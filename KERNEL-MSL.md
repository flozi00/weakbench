# Native Metal kernel development

The `metal-native-v1` evaluator accepts a complete Metal Shading Language compilation unit. Apple’s runtime compiler builds a compute pipeline for the function `evaluate` on an Apple silicon Mac running macOS 13 or later. The trusted launcher requires the Xcode command-line tools. The model code runs as a GPU shader.

```metal
#include <metal_stdlib>
using namespace metal;
kernel void evaluate(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                     device float *out [[buffer(2)]], constant uint4 &dims [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < dims.w) out[i] = a[i] + b[i];
}
```

The dimensions are `M, N, K, L`, with `L` equal to the number of output elements. The host dispatches one-dimensional groups of 64 threads, rounding the grid up to a full group. Each kernel must handle its own final partial group. Identifiers and helper functions are free choices. Workloads may use static threadgroup memory and additional built-in thread indices.

`kernel_msl.compile_native(source)` creates a device pipeline and uses reflection to check the types and access modes of its buffer bindings. Unused inputs may be omitted. The output binding must be present. An additional texture, dynamically allocated threadgroup argument or resource index outside the four-buffer ABI fails this check. Compiled pipeline reports record the GPU name, registry ID and execution width. The compiler field identifies the Apple Metal runtime. A separate field records the version of Clang used to build the Objective-C launcher. [Apple pipeline reflection](https://developer.apple.com/documentation/metal/mtlcomputepipelinereflection/bindings)

`kernel_msl.measure(candidate, baseline, cases, seed=17)` compiles both source strings. For each shape it runs both pipelines, sends float32 inputs, pre-fills the output with NaNs, and compares every output against the stored float64 CPU witness. A finite output passes within `atol + rtol * abs(expected)`. The input buffers and dimensions must retain their original bytes, and the 256-byte guards before and after every buffer must remain intact. A submission must pass every shape before timing. A failed reference makes the evaluation unavailable. Source, build, ABI and verified numerical failures have separate report fields.

Each timed shape uses two warm-up pairs and seven measured pairs with a seeded, balanced order. Every dispatch gets freshly populated, guarded buffers. The runner reads the output back and checks all invariants after each dispatch, including warm-ups and measured pairs. A late failure removes all timing from the report. GPU start/end timestamps cover one command buffer containing one kernel dispatch. CPU transfer, allocation, compilation and numerical verification occur outside that interval. The median of the seven baseline/candidate time ratios is reported with the complete range. [Apple GPU timestamps](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime)

Use the native track on an idle local device. Candidate source is limited to 64,000 UTF-8 bytes, and the input/output inventory to 256 MiB. The source accepts the built-in `<metal_stdlib>` header, ordinary preprocessor definitions and conditionals, and common loop-unrolling pragmas. The trusted launcher runs in a separate process with a private working directory, a reduced environment, a CPU/file-size limit and a wall-clock deadline. These are resource controls. The finite oracle and canaries detect observable output errors, input changes and guard writes. They cannot establish memory safety for arbitrary native shaders. An out-of-bounds read can leave no visible effect, and the macOS driver controls execution of already submitted GPU work. The benchmark’s per-user lock coordinates these measurements with the bounded Metal backend. An inconclusive worker timeout supplies no performance result.

```sh
python3 -BS test_kernel_msl.py
WEAKBENCH_MSL_E2E=1 python3 -BS test_kernel_msl.py
```

The optional speed control compares two native MSL implementations of the sum of an outer product. The expanded form sums every product. The factored form multiplies the two marginal sums. Both use the same public synthetic integer-valued inputs and exact CPU witnesses as the bounded Metal speed control. On an idle Apple GPU the test runs both role assignments on shapes `2049 × 29 × 31` and `3073 × 43 × 47`. Every one of the seven GPU-time pairs for each shape and assignment must identify the factored implementation as faster.

```sh
WEAKBENCH_MSL_SPEED_E2E=1 python3 -BS test_kernel_msl.py
```
