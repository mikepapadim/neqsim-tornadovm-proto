# neqsim-tornadovm-proto

A small standalone prototype for [equinor/neqsim#4587](https://github.com/equinor/neqsim/issues/4587). It runs batches of NeqSim SRK calculations on a GPU with [TornadoVM](https://github.com/beehive-lab/TornadoVM), and compares them against NeqSim itself.

It is a measurement tool, not a NeqSim API. NeqSim is used unchanged, as a Maven dependency.

## What it contains

| Class | Purpose |
|---|---|
| `SrkBatchKernels` | Batched SRK (classic mixing rule): Z and ln φ for both the liquid and vapour roots, in FP64. |
| `SrkFlashKernels` | Batched two-phase TP flash: Wilson K-values, then successive substitution with a Rachford-Rice solve each iteration. |
| `NeqSimSrk` | Builds a NeqSim `SystemSrkEos`, packs its parameters (a, b, m, Tc, Pc, ω, kij), and computes the reference results with NeqSim's own code. |
| `SrkBatchBench`, `SrkFlashBench` | Validation against NeqSim and timing. |

The kernels are plain Java methods over flat `DoubleArray`s, with one state per GPU thread (`KernelContext`). The CPU baseline runs the same methods through a parallel stream.

The fluid is the 12-component gas from the issue. States are random around it:
- fugacities: T 200–500 K, P 1–250 bara
- flash: T 220–400 K, P 5–150 bara
- composition: the feed perturbed log-normally

### Limitations

- The flash has no stability analysis and no acceleration (DEM or Newton), so it is not a replacement for NeqSim's `TPflash`.
- A state whose Rachford-Rice equation has no root in (0, 1) is reported as single phase.
- Only SRK with the classic mixing rule is covered.

## Requirements

- JDK 22 or newer (tested with 25). TornadoVM 8.1.0 `-jdk22plus` is pulled from Maven Central, so no TornadoVM installation is needed.
- One GPU backend, with its driver:
  - **CUDA:** the NVIDIA driver.
  - **OpenCL:** an OpenCL 1.2+ runtime.
  - **Metal:** selectable, but not tested.
- Maven 3.8 or newer.

## Running

```bash
./run.sh srk   validate 2000                     # ln phi and Z vs NeqSim
./run.sh srk   bench 1000 10000 100000 1000000
./run.sh flash validate 20000                    # flash vs NeqSim TPflash
./run.sh flash bench 100000 1000000
BACKEND=opencl ./run.sh flash bench 1000000      # cuda (default), opencl or metal
JAVA_OPTS="-DmaxIt=50" ./run.sh flash bench 1000000
```

## Results

Measured on an RTX 4090 with an i9-13900K (32 threads), JDK 25, TornadoVM 8.1.0 and NeqSim 3.28.0. Each figure is the median of 7–9 warm runs. GPU times include the copies to and from the device.

### Accuracy against NeqSim

The OpenCL and CUDA backends give the same results.

**ln φ and Z**, 2,000 states, 34 of them with three real roots:
- max |Δ ln φ| vs NeqSim: 1.6e-13
- max |ΔZ| vs NeqSim: 3.3e-15
- GPU vs the same code on the CPU: 1.4e-14

**Flash** vs NeqSim `TPflash`, 20,000 states:

| Outcome | States |
|---|---|
| Both two-phase | 16,598 |
| Both single-phase | 3,155 |
| Two-phase in NeqSim only (no stability test in the kernel) | 245 |
| Two-phase in the kernel only | 1 |
| Not converged in 500 iterations | 8 |

In the two-phase states that agree, max |Δβ| is 2.3e-8 and max |Δx, Δy| is 1.4e-8. One near-critical state gives the same split with the phase labels swapped (β ↔ 1−β).

### Timings, 1,000,000 states

| | NeqSim `TPflash`, 32 threads, one system per thread | Kernel on CPU, 32 threads | TornadoVM CUDA | GPU kernel only |
|---|---|---|---|---|
| ln φ + Z, both roots | — | 21 ms | 20 ms | 3.0 ms |
| Two-phase flash | 9.63 s | 1.49 s | 1.66 s | 1.65 s |

- **Fugacities alone are limited by data transfer.** The kernel takes 3 ms, but copying to and from the device takes about 17 ms.
- **On a GeForce card, the flash is limited by FP64 throughput.** Nsight Compute shows the FP64 pipe as the busiest unit (61%), with memory throughput under 2%. The RTX 4090 runs FP64 at 1/64 of its FP32 rate.
  - Capping the iteration count at 50 brings the kernel down to 1.06 s.
  - Changing the memory layout makes no difference.
- The NeqSim column was measured over all 1M states with `JAVA_OPTS="-DneqsimStates=1000000"`. By default the benchmark times NeqSim on 64,000 states and extrapolates.
- **A GPU with full-rate FP64 (A100, H100) is the relevant test.** Those numbers will be added here.

## License

Apache License 2.0.
