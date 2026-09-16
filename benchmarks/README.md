# vext benchmarks

## Run

From the project root:

```bash
python3 benchmarks/run.py
```

This builds the benchmark suite and runs CPU and available CUDA reference measurements.

Useful selections:

```bash
python3 benchmarks/run.py --cpu
python3 benchmarks/run.py --cuda --reduction
python3 benchmarks/run.py --cpu --cuda --unary --axis-reduction
```

Available operation selectors: `--unary`, `--binary`, `--broadcast`, `--logical`, `--reduction`, `--axis-reduction`, `--csr-scatter`, `--csr-spmv`, and `--matmul`.

Quick runs measure each case once and write the ignored `benchmarks/RESULTS.dev.md`.

For paper collection, run 20 independent samples per case and write median/IQR results to `benchmarks/RESULTS.md`:

```bash
python3 benchmarks/run.py --paper --machine-id i5-rtx5060
```

## What is measured

All measurements use FP32.

- CPU kernel baseline: vext versus single-threaded Eigen.
- CUDA kernel baselines: CUB for dense elementwise/reduction operations, cuBLAS SGEMM for matrix multiplication, and cuSPARSE for CSR operations.

Dense unary, binary, broadcast, logical, whole-tensor reduction, and final-axis reduction workloads cover 1D--4D neural-network-style tensors at small, medium, and large scales. Matrix multiplication covers square and rectangular GEMM shapes. CSR scatter and CSR SpMV use rows/degree/features profiles `1024/8/16`, `4096/16/64`, and `16384/32/128`.

CPU kernel timings use Google Benchmark CPU time. CUDA kernel timings use CUDA events. Native kernel measurements exclude allocation, transfers, handles/descriptors, and one-time sparse preprocessing; they represent steady-state kernel time, not end-to-end latency.
