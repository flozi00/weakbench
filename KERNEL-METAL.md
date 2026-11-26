# Apple GPU kernel measurement

The `metal-element-v3` backend compiles a bounded Python kernel body to Metal Shading Language and runs it on the local Apple silicon GPU. The host requires macOS, Xcode command-line tools and the system Metal framework. It builds its Objective-C launcher in a per-user private cache. Model submissions supply expressions and control flow in the following contract.

```json
{
  "threads": 64,
  "outputs_per_thread": 1,
  "source": "total = 0.0\nfor j in range(N):\n    total += A(i * K + j)\nout = total"
}
```

Each logical invocation owns output element `i`. Its unsigned 32-bit dimensions are `M`, `N` and `K`. `L` is the logical output length. The physical threadgroup size is 32, 64, 128 or 256. Each GPU thread owns 1, 2, 4 or 8 consecutive output elements. The launcher guards the tail.

`A(index)` and `B(index)` return Float32 values from the two inputs. Every path assigns `out`. Local assignments and augmented assignments, `if/else`, conditional expressions and ascending finite `for x in range(...)` loops are supported. Start and stop can be nonnegative integer literals, dimensions `M`, `N`, `K`, `L`, or single-assignment local aliases of those values. Aliases must be initialized on every path leading to the loop. The stride is a positive integer literal. A loop index is local to its loop. Local registers must be definitely initialized. Assignments in every path of a loop carry forward when its bounds prove it runs at least once on every evaluated shape. Comments may use Unicode. Identifiers and operations use ASCII. Use `0.0` to initialize a floating accumulator. Integer literals are unsigned except negative scalar literals, which are Float32. Index subtraction wraps modulo 2³². Guard subtraction when handling a lower edge.

The source supports addition, subtraction, multiplication, float division, unsigned integer `//` and `%`, bitwise index operations, comparisons, boolean expressions and `float`, `abs`, `sqrt`, `rsqrt`, `exp`, `min`, `max`, `fma`. Out-of-range reads and integer division by zero invalidate the candidate. The parser accepts only the operations it lowers into Metal. No candidate host code, raw Metal strings, global stores, shared scratchpads or atomics are exposed. The model can change the sequence of computation, read patterns and physical work distribution within this elementwise or loop-reduction contract.

`kernel_metal.measure(candidate, baseline, cases, seed=0)` accepts cases with `shape=[M,N,K]`, flat Float32 input arrays `a` and `b`, a Float64 CPU reference array `expected`, the tolerances `atol` and `rtol`, and optional boolean `timing`. The GPU writes Float32 outputs. The host widens each output to Float64 before comparing it with the stored reference. The case fingerprint includes the full-precision reference bytes. Earlier `metal-element-v2` reports used references rounded to Float32. Every baseline and candidate output must be finite and within `atol + rtol * abs(expected)` term by term. Outcomes distinguish rejected source, numerically incorrect kernels and correct kernels. A missing device, failed host or invalid reference raises `BackendUnavailable`. Measured results include device identity, OS and compiler versions, code and case fingerprints, shape-level checks and performance samples. All validation finishes before performance measurement. The runner checks that the host returned the requested shapes, internally consistent validation counts and seven finite positive time pairs in the scheduled order for each timed shape. An incomplete or inconsistent host report raises `BackendUnavailable`.

On each timing shape, two warmup batches precede seven paired baseline/candidate rounds with seeded, balanced presentation order. Each batch executes the same number of dispatches on both sides. Metal command-buffer GPU start/end timestamps are divided by the batch dispatch count. Timings cover GPU dispatch and device work with resident buffers. The ratio is baseline time divided by candidate time for each pair. The summary gives the median and full spread. `faster` or `slower` means every measured pair has that direction. `overlap` means the direction varies or a pair is equal. The backend serializes WeakBench Metal runs for the operating-system user. Other applications can still use the GPU, and results are specific to the reported device, workload and reference.

```sh
python3 -BS test_kernel_metal.py
WEAKBENCH_METAL_E2E=1 python3 -BS test_kernel_metal.py
```

On an idle GPU the optional speed control runs two mathematically equivalent kernels for a sum of outer products. Their signed integer inputs keep every float32 intermediate exact. One kernel expands each product, and the other factors the sum. The test requires both results to match the CPU oracle on two shapes and all seven paired GPU measurements to identify the faster implementation in both presentation directions.

```sh
WEAKBENCH_METAL_SPEED_E2E=1 python3 -BS test_kernel_metal.py
```
