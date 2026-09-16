#include <benchmark/benchmark.h>

#include <cpu/common.hpp>
#include <cpu/ops/csr.hpp>
#include <cpu/ops/elementwise_binary.hpp>
#include <cpu/ops/elementwise_logical.hpp>
#include <cpu/ops/elementwise_unary.hpp>
#include <cpu/ops/linear_algebra.hpp>
#include <cpu/ops/reduction.hpp>

using vext::benchmarks::cpu::apply_axis_reduction_sizes;
using vext::benchmarks::cpu::apply_csr_scatter_sizes;
using vext::benchmarks::cpu::apply_csr_spmv_sizes;
using vext::benchmarks::cpu::apply_matmul_sizes;
using vext::benchmarks::cpu::apply_matrix_sizes;
using vext::benchmarks::cpu::apply_vector_sizes;
using vext::benchmarks::cpu::CSR_DEGREE;
using vext::benchmarks::cpu::CSR_FEATURES;
using vext::benchmarks::cpu::CSR_ROWS;
using vext::benchmarks::cpu::VECTOR_SIZE;

using vext::benchmarks::cpu::ops::BM_EigenCpuAxisReduction;
using vext::benchmarks::cpu::ops::BM_EigenCpuBinary;
using vext::benchmarks::cpu::ops::BM_EigenCpuBroadcast;
using vext::benchmarks::cpu::ops::BM_EigenCpuCsrScatter;
using vext::benchmarks::cpu::ops::BM_EigenCpuCsrSpmv;
using vext::benchmarks::cpu::ops::BM_EigenCpuLogical;
using vext::benchmarks::cpu::ops::BM_EigenCpuMatmul;
using vext::benchmarks::cpu::ops::BM_EigenCpuReduction;
using vext::benchmarks::cpu::ops::BM_EigenCpuUnary;
using vext::benchmarks::cpu::ops::BM_VextCpuAxisReduction;
using vext::benchmarks::cpu::ops::BM_VextCpuBinary;
using vext::benchmarks::cpu::ops::BM_VextCpuBroadcast;
using vext::benchmarks::cpu::ops::BM_VextCpuCsrScatter;
using vext::benchmarks::cpu::ops::BM_VextCpuCsrSpmv;
using vext::benchmarks::cpu::ops::BM_VextCpuLogical;
using vext::benchmarks::cpu::ops::BM_VextCpuMatmul;
using vext::benchmarks::cpu::ops::BM_VextCpuReduction;
using vext::benchmarks::cpu::ops::BM_VextCpuUnary;

#define REGISTER_CPU_PAIR(FAMILY, OP, SIZE) \
	BENCHMARK_TEMPLATE(BM_VextCpu##FAMILY, ::vext::Op::OP)->Name("Vext/CPU/" #FAMILY "_" #OP)->Apply(apply_vector_sizes)->Iterations(1)

#define REGISTER_CPU_MATRIX_PAIR(FAMILY, OP) \
	BENCHMARK_TEMPLATE(BM_VextCpu##FAMILY, ::vext::Op::OP)->Name("Vext/CPU/" #FAMILY "_" #OP)->Apply(apply_matrix_sizes)->Iterations(1)

#define REGISTER_CPU_AXIS_PAIR(OP) \
	BENCHMARK_TEMPLATE(BM_VextCpuAxisReduction, ::vext::Op::OP)->Name("Vext/CPU/AxisReduction_" #OP)->Apply(apply_axis_reduction_sizes)->Iterations(1)

REGISTER_CPU_PAIR(Unary, ABS, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, SIN, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, COS, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, TANH, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, NEG, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, EXP, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, LOG, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, SQRT, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, SQUARE, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, ROUND, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, SIGMOID, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, SOFT_RELU, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, RELU, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, SOFTMAX, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, SOFTMIN, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, LOGSOFTMAX, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, LEAKY_RELU, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, ELU, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, SWISH, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, LINEAR, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, CLIP, VECTOR_SIZE);
REGISTER_CPU_PAIR(Unary, POW, VECTOR_SIZE);

REGISTER_CPU_PAIR(Binary, ADD, VECTOR_SIZE);
REGISTER_CPU_PAIR(Binary, SUB, VECTOR_SIZE);
REGISTER_CPU_PAIR(Binary, MUL, VECTOR_SIZE);
REGISTER_CPU_PAIR(Binary, DIV, VECTOR_SIZE);
REGISTER_CPU_PAIR(Binary, POW, VECTOR_SIZE);
REGISTER_CPU_PAIR(Binary, MIN, VECTOR_SIZE);
REGISTER_CPU_PAIR(Binary, MAX, VECTOR_SIZE);
REGISTER_CPU_PAIR(Binary, PRELU, VECTOR_SIZE);

REGISTER_CPU_MATRIX_PAIR(Broadcast, ADD);
REGISTER_CPU_MATRIX_PAIR(Broadcast, SUB);
REGISTER_CPU_MATRIX_PAIR(Broadcast, MUL);
REGISTER_CPU_MATRIX_PAIR(Broadcast, DIV);
REGISTER_CPU_MATRIX_PAIR(Broadcast, POW);
REGISTER_CPU_MATRIX_PAIR(Broadcast, MIN);
REGISTER_CPU_MATRIX_PAIR(Broadcast, MAX);
REGISTER_CPU_MATRIX_PAIR(Broadcast, PRELU);

REGISTER_CPU_PAIR(Logical, EQUAL, VECTOR_SIZE);
REGISTER_CPU_PAIR(Logical, NOT_EQUAL, VECTOR_SIZE);
REGISTER_CPU_PAIR(Logical, LESS, VECTOR_SIZE);
REGISTER_CPU_PAIR(Logical, LESS_EQUAL, VECTOR_SIZE);
REGISTER_CPU_PAIR(Logical, GREATER, VECTOR_SIZE);
REGISTER_CPU_PAIR(Logical, GREATER_EQUAL, VECTOR_SIZE);

REGISTER_CPU_PAIR(Reduction, SUM, VECTOR_SIZE);
REGISTER_CPU_PAIR(Reduction, MEAN, VECTOR_SIZE);
REGISTER_CPU_PAIR(Reduction, MIN, VECTOR_SIZE);
REGISTER_CPU_PAIR(Reduction, MAX, VECTOR_SIZE);
REGISTER_CPU_PAIR(Reduction, PROD, VECTOR_SIZE);
REGISTER_CPU_PAIR(Reduction, STD, VECTOR_SIZE);
REGISTER_CPU_PAIR(Reduction, VAR, VECTOR_SIZE);
REGISTER_CPU_PAIR(Reduction, L2_NORM, VECTOR_SIZE);

REGISTER_CPU_AXIS_PAIR(SUM);
REGISTER_CPU_AXIS_PAIR(MEAN);
REGISTER_CPU_AXIS_PAIR(MIN);
REGISTER_CPU_AXIS_PAIR(MAX);
REGISTER_CPU_AXIS_PAIR(PROD);
REGISTER_CPU_AXIS_PAIR(STD);
REGISTER_CPU_AXIS_PAIR(VAR);
REGISTER_CPU_AXIS_PAIR(L2_NORM);

BENCHMARK_TEMPLATE(BM_VextCpuCsrScatter, ::vext::Op::SUM)->Name("Vext/CPU/CSRScatter_SUM")->Apply(apply_csr_scatter_sizes)->Iterations(1);
BENCHMARK_TEMPLATE(BM_VextCpuCsrScatter, ::vext::Op::MEAN)->Name("Vext/CPU/CSRScatter_MEAN")->Apply(apply_csr_scatter_sizes)->Iterations(1);
BENCHMARK_TEMPLATE(BM_VextCpuCsrSpmv, ::vext::Op::SUM)->Name("Vext/CPU/CSRSpMV_SUM")->Apply(apply_csr_spmv_sizes)->Iterations(1);
BENCHMARK_TEMPLATE(BM_VextCpuCsrSpmv, ::vext::Op::MEAN)->Name("Vext/CPU/CSRSpMV_MEAN")->Apply(apply_csr_spmv_sizes)->Iterations(1);

BENCHMARK(BM_VextCpuMatmul)->Name("Vext/CPU/Matmul")->Apply(apply_matmul_sizes)->Iterations(1);

#define REGISTER_CPU_REFERENCE(FAMILY, OP, SIZE) \
	BENCHMARK_TEMPLATE(BM_EigenCpu##FAMILY, ::vext::Op::OP)->Name("Eigen/CPU/" #FAMILY "_" #OP)->Apply(apply_vector_sizes)->Iterations(1)

#define REGISTER_CPU_MATRIX_REFERENCE(FAMILY, OP) \
	BENCHMARK_TEMPLATE(BM_EigenCpu##FAMILY, ::vext::Op::OP)->Name("Eigen/CPU/" #FAMILY "_" #OP)->Apply(apply_matrix_sizes)->Iterations(1)

#define REGISTER_CPU_AXIS_REFERENCE(OP) \
	BENCHMARK_TEMPLATE(BM_EigenCpuAxisReduction, ::vext::Op::OP)->Name("Eigen/CPU/AxisReduction_" #OP)->Apply(apply_axis_reduction_sizes)->Iterations(1)

REGISTER_CPU_REFERENCE(Unary, ABS, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, SIN, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, COS, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, TANH, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, NEG, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, EXP, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, LOG, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, SQRT, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, SQUARE, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, ROUND, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, SIGMOID, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, SOFT_RELU, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, RELU, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, SOFTMAX, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, SOFTMIN, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, LOGSOFTMAX, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, LEAKY_RELU, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, ELU, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, SWISH, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, LINEAR, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, CLIP, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Unary, POW, VECTOR_SIZE);

REGISTER_CPU_REFERENCE(Binary, ADD, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Binary, SUB, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Binary, MUL, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Binary, DIV, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Binary, POW, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Binary, MIN, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Binary, MAX, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Binary, PRELU, VECTOR_SIZE);

REGISTER_CPU_MATRIX_REFERENCE(Broadcast, ADD);
REGISTER_CPU_MATRIX_REFERENCE(Broadcast, SUB);
REGISTER_CPU_MATRIX_REFERENCE(Broadcast, MUL);
REGISTER_CPU_MATRIX_REFERENCE(Broadcast, DIV);
REGISTER_CPU_MATRIX_REFERENCE(Broadcast, POW);
REGISTER_CPU_MATRIX_REFERENCE(Broadcast, MIN);
REGISTER_CPU_MATRIX_REFERENCE(Broadcast, MAX);
REGISTER_CPU_MATRIX_REFERENCE(Broadcast, PRELU);

REGISTER_CPU_REFERENCE(Logical, EQUAL, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Logical, NOT_EQUAL, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Logical, LESS, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Logical, LESS_EQUAL, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Logical, GREATER, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Logical, GREATER_EQUAL, VECTOR_SIZE);

REGISTER_CPU_REFERENCE(Reduction, SUM, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Reduction, MEAN, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Reduction, MIN, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Reduction, MAX, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Reduction, PROD, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Reduction, STD, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Reduction, VAR, VECTOR_SIZE);
REGISTER_CPU_REFERENCE(Reduction, L2_NORM, VECTOR_SIZE);

REGISTER_CPU_AXIS_REFERENCE(SUM);
REGISTER_CPU_AXIS_REFERENCE(MEAN);
REGISTER_CPU_AXIS_REFERENCE(MIN);
REGISTER_CPU_AXIS_REFERENCE(MAX);
REGISTER_CPU_AXIS_REFERENCE(PROD);
REGISTER_CPU_AXIS_REFERENCE(STD);
REGISTER_CPU_AXIS_REFERENCE(VAR);
REGISTER_CPU_AXIS_REFERENCE(L2_NORM);

BENCHMARK_TEMPLATE(BM_EigenCpuCsrScatter, ::vext::Op::SUM)->Name("Eigen/CPU/CSRScatter_SUM")->Apply(apply_csr_scatter_sizes)->Iterations(1);
BENCHMARK_TEMPLATE(BM_EigenCpuCsrScatter, ::vext::Op::MEAN)->Name("Eigen/CPU/CSRScatter_MEAN")->Apply(apply_csr_scatter_sizes)->Iterations(1);
BENCHMARK_TEMPLATE(BM_EigenCpuCsrSpmv, ::vext::Op::SUM)->Name("Eigen/CPU/CSRSpMV_SUM")->Apply(apply_csr_spmv_sizes)->Iterations(1);
BENCHMARK_TEMPLATE(BM_EigenCpuCsrSpmv, ::vext::Op::MEAN)->Name("Eigen/CPU/CSRSpMV_MEAN")->Apply(apply_csr_spmv_sizes)->Iterations(1);

BENCHMARK(BM_EigenCpuMatmul)->Name("Eigen/CPU/Matmul")->Apply(apply_matmul_sizes)->Iterations(1);

#undef REGISTER_CPU_MATRIX_REFERENCE
#undef REGISTER_CPU_AXIS_REFERENCE
#undef REGISTER_CPU_REFERENCE
#undef REGISTER_CPU_MATRIX_PAIR
#undef REGISTER_CPU_AXIS_PAIR
#undef REGISTER_CPU_PAIR

BENCHMARK_MAIN();
