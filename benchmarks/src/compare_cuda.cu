#include <benchmark/benchmark.h>

#include <cuda/common.cuh>
#include <cuda/ops/csr.cuh>
#include <cuda/ops/elementwise_binary.cuh>
#include <cuda/ops/elementwise_logical.cuh>
#include <cuda/ops/elementwise_unary.cuh>
#include <cuda/ops/linear_algebra.cuh>
#include <cuda/ops/reduction.cuh>

using vext::benchmarks::cuda::apply_axis_reduction_sizes;
using vext::benchmarks::cuda::apply_csr_scatter_sizes;
using vext::benchmarks::cuda::apply_csr_spmv_sizes;
using vext::benchmarks::cuda::apply_matmul_sizes;
using vext::benchmarks::cuda::apply_matrix_sizes;
using vext::benchmarks::cuda::apply_vector_sizes;
using vext::benchmarks::cuda::CSR_DEGREE;
using vext::benchmarks::cuda::CSR_FEATURES;
using vext::benchmarks::cuda::CSR_ROWS;
using vext::benchmarks::cuda::VECTOR_SIZE;

using vext::benchmarks::cuda::ops::BM_CubCudaAxisReduction;
using vext::benchmarks::cuda::ops::BM_CubCudaBinary;
using vext::benchmarks::cuda::ops::BM_CubCudaBroadcast;
using vext::benchmarks::cuda::ops::BM_CubCudaLogical;
using vext::benchmarks::cuda::ops::BM_CubCudaReduction;
using vext::benchmarks::cuda::ops::BM_CubCudaUnary;
using vext::benchmarks::cuda::ops::BM_CublasCudaMatmul;
using vext::benchmarks::cuda::ops::BM_CusparseCudaCsrScatter;
using vext::benchmarks::cuda::ops::BM_CusparseCudaCsrSpmv;
using vext::benchmarks::cuda::ops::BM_VextCudaAxisReduction;
using vext::benchmarks::cuda::ops::BM_VextCudaBinary;
using vext::benchmarks::cuda::ops::BM_VextCudaBroadcast;
using vext::benchmarks::cuda::ops::BM_VextCudaCsrScatter;
using vext::benchmarks::cuda::ops::BM_VextCudaCsrSpmv;
using vext::benchmarks::cuda::ops::BM_VextCudaLogical;
using vext::benchmarks::cuda::ops::BM_VextCudaMatmul;
using vext::benchmarks::cuda::ops::BM_VextCudaReduction;
using vext::benchmarks::cuda::ops::BM_VextCudaUnary;

#define REGISTER_CUDA_PAIR(FAMILY, OP, SIZE) \
	BENCHMARK_TEMPLATE(BM_VextCuda##FAMILY, ::vext::Op::OP)->Name("Vext/CUDA/" #FAMILY "_" #OP)->Apply(apply_vector_sizes)->Iterations(1)->UseManualTime()

#define REGISTER_CUDA_MATRIX_PAIR(FAMILY, OP) \
	BENCHMARK_TEMPLATE(BM_VextCuda##FAMILY, ::vext::Op::OP)->Name("Vext/CUDA/" #FAMILY "_" #OP)->Apply(apply_matrix_sizes)->Iterations(1)->UseManualTime()

#define REGISTER_CUDA_AXIS_PAIR(OP) \
	BENCHMARK_TEMPLATE(BM_VextCudaAxisReduction, ::vext::Op::OP)->Name("Vext/CUDA/AxisReduction_" #OP)->Apply(apply_axis_reduction_sizes)->Iterations(1)->UseManualTime()

REGISTER_CUDA_PAIR(Unary, ABS, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, SIN, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, COS, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, TANH, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, NEG, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, EXP, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, LOG, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, SQRT, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, SQUARE, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, ROUND, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, SIGMOID, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, SOFT_RELU, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, RELU, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, SOFTMAX, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, SOFTMIN, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, LOGSOFTMAX, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, LEAKY_RELU, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, ELU, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, SWISH, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, LINEAR, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, CLIP, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, POW, VECTOR_SIZE);

REGISTER_CUDA_PAIR(Binary, ADD, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Binary, SUB, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Binary, MUL, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Binary, DIV, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Binary, POW, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Binary, MIN, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Binary, MAX, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Binary, PRELU, VECTOR_SIZE);

REGISTER_CUDA_MATRIX_PAIR(Broadcast, ADD);
REGISTER_CUDA_MATRIX_PAIR(Broadcast, SUB);
REGISTER_CUDA_MATRIX_PAIR(Broadcast, MUL);
REGISTER_CUDA_MATRIX_PAIR(Broadcast, DIV);
REGISTER_CUDA_MATRIX_PAIR(Broadcast, POW);
REGISTER_CUDA_MATRIX_PAIR(Broadcast, MIN);
REGISTER_CUDA_MATRIX_PAIR(Broadcast, MAX);
REGISTER_CUDA_MATRIX_PAIR(Broadcast, PRELU);

REGISTER_CUDA_PAIR(Logical, EQUAL, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Logical, NOT_EQUAL, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Logical, LESS, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Logical, LESS_EQUAL, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Logical, GREATER, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Logical, GREATER_EQUAL, VECTOR_SIZE);

REGISTER_CUDA_PAIR(Reduction, SUM, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Reduction, MEAN, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Reduction, MIN, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Reduction, MAX, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Reduction, PROD, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Reduction, STD, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Reduction, VAR, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Reduction, L2_NORM, VECTOR_SIZE);

REGISTER_CUDA_AXIS_PAIR(SUM);
REGISTER_CUDA_AXIS_PAIR(MEAN);
REGISTER_CUDA_AXIS_PAIR(MIN);
REGISTER_CUDA_AXIS_PAIR(MAX);
REGISTER_CUDA_AXIS_PAIR(PROD);
REGISTER_CUDA_AXIS_PAIR(STD);
REGISTER_CUDA_AXIS_PAIR(VAR);
REGISTER_CUDA_AXIS_PAIR(L2_NORM);

BENCHMARK_TEMPLATE(BM_VextCudaCsrScatter, ::vext::Op::SUM)->Name("Vext/CUDA/CSRScatter_SUM")->Apply(apply_csr_scatter_sizes)->Iterations(1)->UseManualTime();
BENCHMARK_TEMPLATE(BM_VextCudaCsrScatter, ::vext::Op::MEAN)->Name("Vext/CUDA/CSRScatter_MEAN")->Apply(apply_csr_scatter_sizes)->Iterations(1)->UseManualTime();
BENCHMARK_TEMPLATE(BM_VextCudaCsrSpmv, ::vext::Op::SUM)->Name("Vext/CUDA/CSRSpMV_SUM")->Apply(apply_csr_spmv_sizes)->Iterations(1)->UseManualTime();
BENCHMARK_TEMPLATE(BM_VextCudaCsrSpmv, ::vext::Op::MEAN)->Name("Vext/CUDA/CSRSpMV_MEAN")->Apply(apply_csr_spmv_sizes)->Iterations(1)->UseManualTime();

BENCHMARK(BM_VextCudaMatmul)->Name("Vext/CUDA/Matmul")->Apply(apply_matmul_sizes)->Iterations(1)->UseManualTime();

#define REGISTER_CUDA_REFERENCE(FAMILY, OP, SIZE) \
	BENCHMARK_TEMPLATE(BM_CubCuda##FAMILY, ::vext::Op::OP)->Name("CUB/CUDA/" #FAMILY "_" #OP)->Apply(apply_vector_sizes)->Iterations(1)->UseManualTime()

#define REGISTER_CUDA_MATRIX_REFERENCE(FAMILY, OP) \
	BENCHMARK_TEMPLATE(BM_CubCuda##FAMILY, ::vext::Op::OP)->Name("CUB/CUDA/" #FAMILY "_" #OP)->Apply(apply_matrix_sizes)->Iterations(1)->UseManualTime()

#define REGISTER_CUDA_AXIS_REFERENCE(OP) \
	BENCHMARK_TEMPLATE(BM_CubCudaAxisReduction, ::vext::Op::OP)->Name("CUB/CUDA/AxisReduction_" #OP)->Apply(apply_axis_reduction_sizes)->Iterations(1)->UseManualTime()

REGISTER_CUDA_REFERENCE(Unary, ABS, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, SIN, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, COS, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, TANH, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, NEG, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, EXP, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, LOG, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, SQRT, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, SQUARE, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, ROUND, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, SIGMOID, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, SOFT_RELU, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, RELU, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, SOFTMAX, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, SOFTMIN, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, LOGSOFTMAX, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, LEAKY_RELU, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, ELU, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, SWISH, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, LINEAR, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, CLIP, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Unary, POW, VECTOR_SIZE);

REGISTER_CUDA_REFERENCE(Binary, ADD, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Binary, SUB, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Binary, MUL, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Binary, DIV, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Binary, POW, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Binary, MIN, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Binary, MAX, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Binary, PRELU, VECTOR_SIZE);

REGISTER_CUDA_MATRIX_REFERENCE(Broadcast, ADD);
REGISTER_CUDA_MATRIX_REFERENCE(Broadcast, SUB);
REGISTER_CUDA_MATRIX_REFERENCE(Broadcast, MUL);
REGISTER_CUDA_MATRIX_REFERENCE(Broadcast, DIV);
REGISTER_CUDA_MATRIX_REFERENCE(Broadcast, POW);
REGISTER_CUDA_MATRIX_REFERENCE(Broadcast, MIN);
REGISTER_CUDA_MATRIX_REFERENCE(Broadcast, MAX);
REGISTER_CUDA_MATRIX_REFERENCE(Broadcast, PRELU);

REGISTER_CUDA_REFERENCE(Logical, EQUAL, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Logical, NOT_EQUAL, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Logical, LESS, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Logical, LESS_EQUAL, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Logical, GREATER, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Logical, GREATER_EQUAL, VECTOR_SIZE);

REGISTER_CUDA_REFERENCE(Reduction, SUM, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Reduction, MEAN, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Reduction, MIN, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Reduction, MAX, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Reduction, PROD, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Reduction, STD, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Reduction, VAR, VECTOR_SIZE);
REGISTER_CUDA_REFERENCE(Reduction, L2_NORM, VECTOR_SIZE);

REGISTER_CUDA_AXIS_REFERENCE(SUM);
REGISTER_CUDA_AXIS_REFERENCE(MEAN);
REGISTER_CUDA_AXIS_REFERENCE(MIN);
REGISTER_CUDA_AXIS_REFERENCE(MAX);
REGISTER_CUDA_AXIS_REFERENCE(PROD);
REGISTER_CUDA_AXIS_REFERENCE(STD);
REGISTER_CUDA_AXIS_REFERENCE(VAR);
REGISTER_CUDA_AXIS_REFERENCE(L2_NORM);

BENCHMARK_TEMPLATE(BM_CusparseCudaCsrScatter, ::vext::Op::SUM)->Name("cuSPARSE/CUDA/CSRScatter_SUM")->Apply(apply_csr_scatter_sizes)->Iterations(1)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CusparseCudaCsrScatter, ::vext::Op::MEAN)->Name("cuSPARSE/CUDA/CSRScatter_MEAN")->Apply(apply_csr_scatter_sizes)->Iterations(1)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CusparseCudaCsrSpmv, ::vext::Op::SUM)->Name("cuSPARSE/CUDA/CSRSpMV_SUM")->Apply(apply_csr_spmv_sizes)->Iterations(1)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CusparseCudaCsrSpmv, ::vext::Op::MEAN)->Name("cuSPARSE/CUDA/CSRSpMV_MEAN")->Apply(apply_csr_spmv_sizes)->Iterations(1)->UseManualTime();

BENCHMARK(BM_CublasCudaMatmul)->Name("cuBLAS/CUDA/Matmul")->Apply(apply_matmul_sizes)->Iterations(1)->UseManualTime();

#undef REGISTER_CUDA_MATRIX_REFERENCE
#undef REGISTER_CUDA_AXIS_REFERENCE
#undef REGISTER_CUDA_REFERENCE
#undef REGISTER_CUDA_MATRIX_PAIR
#undef REGISTER_CUDA_AXIS_PAIR
#undef REGISTER_CUDA_PAIR

BENCHMARK_MAIN();
