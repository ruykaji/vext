# vext benchmarks

The benchmark project is standalone and compares vext with Eigen on CPU and NVIDIA Thrust, cuBLAS, and cuSPARSE on CUDA. Eigen and Google Benchmark are resolved through Conan; CUDA reference libraries are provided by the installed CUDA toolkit.

The paired suite covers dense unary, binary, broadcast, logical, reduction, and matrix multiplication operations. Standard CSR scatter and CSR matrix-vector multiplication cases use Eigen Sparse on CPU and cuSPARSE on CUDA. All benchmark inputs are either 1D vectors or 2D matrices, and Eigen runs single-threaded.

## Requirements

- CMake 3.24 or newer
- Conan 2
- Ninja
- A C++20 compiler
- CUDA toolkit and an NVIDIA GPU for CUDA benchmarks

## Build and run

From the project root, run:

```bash
python3 benchmarks/run.py
```

The script installs benchmark dependencies, builds vext, builds the standalone benchmark project, and runs the comparison suite.

To run only CPU benchmarks:

```bash
python3 benchmarks/run.py --cuda off
```

To require CUDA benchmarks:

```bash
python3 benchmarks/run.py --cuda on
```
