# Native CUDA kernel evaluator

WeakBench accepts a JSON object with standalone NVRTC CUDA C++ in `source`, a block size of 32, 64, 128 or 256 in `threads`, and 1, 2, 4 or 8 in `outputs_per_thread`. The submitted kernel supplies this entry point and handles its own final-block bounds.

```cpp
extern "C" __global__ void evaluate(const float* a, const float* b, float* out,
                                    unsigned M, unsigned N, unsigned K, unsigned L);
```

The runner dispatches a one-dimensional grid of `ceil(L / (threads * outputs_per_thread))` blocks. `a`, `b` and `out` are disjoint GPU buffers. All tasks use FP32 inputs and outputs. The operator supplies each case's `[M,N,K]`, input arrays, finite CPU oracle, absolute and relative tolerances, and a boolean `timing` flag. Source has access to NVRTC built-ins. Host wrappers and external libraries are outside this ABI.

Build once from the WeakBench root on a local Linux Docker daemon (amd64 or arm64). The image build fetches a digest-pinned NVRTC 13.0.88 wheel. Each evaluation runs in a separate read-only, network-disabled, resource-bounded container as UID 65534. The runner sends input over stdin and mounts no workspace directory.

```sh
docker build -t weakbench-kernel-cuda:1 kernel-cuda
WEAKBENCH_CUDA_COMPILE_E2E=1 python3 -m unittest -v test_kernel_cuda
```

`kernel_cuda.compile_native(program)` emits a SHA-256 hash and byte count after compiling native SM120 SASS and checking for an executable `evaluate` entry in the CUBIN. The loader tries `WEAKBENCH_KERNEL_DOCKER_SOCKET`, then `WEAKBENCH_ARENA_DOCKER_SOCKET`, then a local Unix `DOCKER_HOST` or `/var/run/docker.sock`. Compilation works on machines without a GPU. [NVIDIA NVRTC](https://docs.nvidia.com/cuda/archive/13.0.2/nvrtc/index.html)

Hardware measurement requires a dedicated local NVIDIA SM120 device and a working NVIDIA Container Toolkit. Set `WEAKBENCH_CUDA_DEVICE` to that card's full `GPU-…` UUID. After building the image, run the hardware control with the chosen card before using kernel scores.

```sh
WEAKBENCH_CUDA_HARDWARE_E2E=1 python3 -m unittest -v test_kernel_cuda.HardwareControl
```

`kernel_cuda.measure(candidate, baseline, cases, seed=0)` takes an exclusive local lock per selected UUID. The driver verifies the visible device count, UUID and compute capability. Both kernels run against the CPU oracle on every shape. The evaluator loads a fresh module for each dispatch to reset device globals between trials. The runner checks input immutability, red zones, unwritten outputs, finite values and `abs(error) <= atol + rtol * abs(expected)` on every dispatch. A failed candidate suppresses timing for the whole result. An invalid baseline or unavailable backend raises `BackendUnavailable`.

For each configured timing shape, the evaluator warms both kernels and records seven pairs with balanced, seeded order on the same device. Each sample encloses a single dispatch between external CUDA event nodes within one GPU graph. The report includes each duration in microseconds, the median of paired speed ratios, the range of ratios, both CUBIN hashes, device UUID, driver version and immutable evaluator image ID. These ratios compare the candidate with the supplied baseline for that shape. [CUDA Driver event API](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-driver-api/group__CUDA__EVENT.html)
