#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <vext/ops.hpp>
#include <vext/tensor.hpp>

namespace
{

constexpr std::int64_t VECTOR_SIZE   = 1 << 20;
constexpr std::int64_t MATRIX_ROWS   = 1024;
constexpr std::int64_t MATRIX_COLS   = 1024;
constexpr std::int64_t CSR_ROWS      = 4096;
constexpr std::int64_t CSR_DEGREE    = 16;
constexpr std::int64_t CSR_FEATURES  = 64;
constexpr float        UNARY_PARAM_A = 1.25f;
constexpr float        UNARY_PARAM_B = 2.0f;

std::vector<float>
make_values(
	const std::int64_t size,
	const bool         positive = false)
{
	std::vector<float> values(static_cast<std::size_t>(size));

	for(std::int64_t i = 0; i < size; ++i)
		{
			const float value                   = static_cast<float>(i % 1024) / 256.0f - 2.0f;
			values[static_cast<std::size_t>(i)] = positive ? std::abs(value) + 0.25f : value;
		}

	return values;
}

template <vext::Op Kp>
std::vector<float>
make_unary_values(
	const std::int64_t size)
{
	return make_values(size, Kp == vext::Op::LOG || Kp == vext::Op::SQRT || Kp == vext::Op::POW);
}

template <vext::Op Kp>
void
run_vext_unary(
	vext::Tensor<float>& values)
{
	if constexpr(Kp == vext::Op::LEAKY_RELU || Kp == vext::Op::ELU || Kp == vext::Op::SWISH)
		{
			vext::unary<Kp>(values, UNARY_PARAM_A);
		}
	else if constexpr(Kp == vext::Op::LINEAR || Kp == vext::Op::CLIP || Kp == vext::Op::POW)
		{
			vext::unary<Kp>(values, UNARY_PARAM_A, UNARY_PARAM_B);
		}
	else
		{
			vext::unary<Kp>(values);
		}
}

template <vext::Op Kp>
void
run_eigen_unary(
	Eigen::ArrayXf& values)
{
	if constexpr(Kp == vext::Op::ABS)
		{
			values = values.abs();
		}
	else if constexpr(Kp == vext::Op::SIN)
		{
			values = values.sin();
		}
	else if constexpr(Kp == vext::Op::COS)
		{
			values = values.cos();
		}
	else if constexpr(Kp == vext::Op::TANH)
		{
			values = values.tanh();
		}
	else if constexpr(Kp == vext::Op::NEG)
		{
			values = -values;
		}
	else if constexpr(Kp == vext::Op::EXP)
		{
			values = values.exp();
		}
	else if constexpr(Kp == vext::Op::LOG)
		{
			values = values.log();
		}
	else if constexpr(Kp == vext::Op::SQRT)
		{
			values = values.sqrt();
		}
	else if constexpr(Kp == vext::Op::SQUARE)
		{
			values = values.square();
		}
	else if constexpr(Kp == vext::Op::ROUND)
		{
			values = values.round();
		}
	else if constexpr(Kp == vext::Op::SIGMOID)
		{
			values = 1.0f / (1.0f + (-values).exp());
		}
	else if constexpr(Kp == vext::Op::SOFT_RELU)
		{
			values = (1.0f + values.exp()).log();
		}
	else if constexpr(Kp == vext::Op::RELU)
		{
			values = values.max(0.0f);
		}
	else if constexpr(Kp == vext::Op::SOFTMAX)
		{
			values = values.exp();
			values /= values.sum();
		}
	else if constexpr(Kp == vext::Op::SOFTMIN)
		{
			values = (-values).exp();
			values /= values.sum();
		}
	else if constexpr(Kp == vext::Op::LOGSOFTMAX)
		{
			values = values.exp();
			values = (values / values.sum()).log();
		}
	else if constexpr(Kp == vext::Op::LEAKY_RELU)
		{
			values = values.max(0.0f) + UNARY_PARAM_A * values.min(0.0f);
		}
	else if constexpr(Kp == vext::Op::ELU)
		{
			values = (values > 0.0f).select(values, UNARY_PARAM_A * (values.exp() - 1.0f));
		}
	else if constexpr(Kp == vext::Op::SWISH)
		{
			values = values / (1.0f + (-UNARY_PARAM_A * values).exp());
		}
	else if constexpr(Kp == vext::Op::LINEAR)
		{
			values = UNARY_PARAM_A * values + UNARY_PARAM_B;
		}
	else if constexpr(Kp == vext::Op::CLIP)
		{
			values = values.max(UNARY_PARAM_A).min(UNARY_PARAM_B);
		}
	else if constexpr(Kp == vext::Op::POW)
		{
			values = UNARY_PARAM_A * values.pow(UNARY_PARAM_B);
		}
}

template <vext::Op Kp>
void
BM_VextCpuUnary(
	benchmark::State& state)
{
	const std::uint32_t size  = static_cast<std::uint32_t>(state.range(0));
	const auto          input = make_unary_values<Kp>(size);

	vext::Tensor<float> values(size);
	values.set_from(input);

	for([[maybe_unused]] auto iteration : state)
		{
			run_vext_unary<Kp>(values);
			benchmark::DoNotOptimize(values.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <vext::Op Kp>
void
BM_EigenCpuUnary(
	benchmark::State& state)
{
	const Eigen::Index                     size         = state.range(0);
	const auto                             input_values = make_unary_values<Kp>(size);
	const Eigen::Map<const Eigen::ArrayXf> input(input_values.data(), size);

	Eigen::ArrayXf values = input;

	for([[maybe_unused]] auto iteration : state)
		{
			run_eigen_unary<Kp>(values);
			benchmark::DoNotOptimize(values.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <vext::Op Kp, typename Out, typename Lhs, typename Rhs>
void
run_eigen_binary(
	Out&       out,
	const Lhs& lhs,
	const Rhs& rhs)
{
	if constexpr(Kp == vext::Op::ADD)
		{
			out = lhs + rhs;
		}
	else if constexpr(Kp == vext::Op::SUB)
		{
			out = lhs - rhs;
		}
	else if constexpr(Kp == vext::Op::MUL)
		{
			out = lhs * rhs;
		}
	else if constexpr(Kp == vext::Op::DIV)
		{
			out = lhs / rhs;
		}
	else if constexpr(Kp == vext::Op::POW)
		{
			out = lhs.pow(rhs);
		}
	else if constexpr(Kp == vext::Op::MIN)
		{
			out = lhs.min(rhs);
		}
	else if constexpr(Kp == vext::Op::MAX)
		{
			out = lhs.max(rhs);
		}
	else
		{
			out = lhs.max(0.0f) + rhs * lhs.min(0.0f);
		}
}

template <vext::Op Kp>
void
BM_VextCpuBinary(
	benchmark::State& state)
{
	const std::uint32_t size = static_cast<std::uint32_t>(state.range(0));

	vext::Tensor<float> lhs(size);
	vext::Tensor<float> rhs(size);
	vext::Tensor<float> out(size);

	lhs.set_from(make_values(size, Kp == vext::Op::POW));
	rhs.set_from(make_values(size, true));

	for([[maybe_unused]] auto iteration : state)
		{
			vext::binary<Kp>(lhs, rhs, out);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <vext::Op Kp>
void
BM_EigenCpuBinary(
	benchmark::State& state)
{
	const Eigen::Index                     size       = state.range(0);
	const auto                             lhs_values = make_values(size, Kp == vext::Op::POW);
	const auto                             rhs_values = make_values(size, true);
	const Eigen::Map<const Eigen::ArrayXf> lhs(lhs_values.data(), size);
	const Eigen::Map<const Eigen::ArrayXf> rhs(rhs_values.data(), size);

	Eigen::ArrayXf out(size);

	for([[maybe_unused]] auto iteration : state)
		{
			run_eigen_binary<Kp>(out, lhs, rhs);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <vext::Op Kp>
void
BM_VextCpuBroadcast(
	benchmark::State& state)
{
	const std::uint32_t rows = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t cols = static_cast<std::uint32_t>(state.range(1));

	vext::Tensor<float> matrix(rows, cols);
	vext::Tensor<float> row(cols);
	vext::Tensor<float> out(rows, cols);

	matrix.set_from(make_values(static_cast<std::int64_t>(rows) * cols, Kp == vext::Op::POW));
	row.set_from(make_values(cols, true));

	for([[maybe_unused]] auto iteration : state)
		{
			vext::binary<Kp>(matrix, row, out);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * rows * cols);
}

template <vext::Op Kp>
void
BM_EigenCpuBroadcast(
	benchmark::State& state)
{
	const Eigen::Index rows          = state.range(0);
	const Eigen::Index cols          = state.range(1);
	const auto         matrix_values = make_values(rows * cols, Kp == vext::Op::POW);
	const auto         row_values    = make_values(cols, true);

	const Eigen::Map<const Eigen::ArrayXXf> matrix(matrix_values.data(), cols, rows);
	const Eigen::Map<const Eigen::ArrayXf>  row(row_values.data(), cols);

	Eigen::ArrayXXf out(cols, rows);

	for([[maybe_unused]] auto iteration : state)
		{
			run_eigen_binary<Kp>(out, matrix, row.replicate(1, rows));
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * rows * cols);
}

template <vext::Op Kp>
void
BM_VextCpuLogical(
	benchmark::State& state)
{
	const std::uint32_t size = static_cast<std::uint32_t>(state.range(0));

	vext::Tensor<float>        lhs(size);
	vext::Tensor<float>        rhs(size);
	vext::Tensor<std::uint8_t> out(size);

	lhs.set_from(make_values(size));
	rhs.set_from(make_values(size, true));

	for([[maybe_unused]] auto iteration : state)
		{
			vext::logical<Kp>(lhs, rhs, out);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <vext::Op Kp>
void
BM_EigenCpuLogical(
	benchmark::State& state)
{
	const Eigen::Index size       = state.range(0);
	const auto         lhs_values = make_values(size);
	const auto         rhs_values = make_values(size, true);

	const Eigen::Map<const Eigen::ArrayXf> lhs(lhs_values.data(), size);
	const Eigen::Map<const Eigen::ArrayXf> rhs(rhs_values.data(), size);

	Eigen::Array<bool, Eigen::Dynamic, 1> out(size);

	for([[maybe_unused]] auto iteration : state)
		{
			if constexpr(Kp == vext::Op::EQUAL)
				{
					out = lhs == rhs;
				}
			else if constexpr(Kp == vext::Op::NOT_EQUAL)
				{
					out = lhs != rhs;
				}
			else if constexpr(Kp == vext::Op::LESS)
				{
					out = lhs < rhs;
				}
			else if constexpr(Kp == vext::Op::LESS_EQUAL)
				{
					out = lhs <= rhs;
				}
			else if constexpr(Kp == vext::Op::GREATER)
				{
					out = lhs > rhs;
				}
			else
				{
					out = lhs >= rhs;
				}

			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <vext::Op Kp>
float
eigen_reduce(
	const Eigen::ArrayXf& values)
{
	if constexpr(Kp == vext::Op::SUM)
		{
			return values.sum();
		}
	else if constexpr(Kp == vext::Op::MEAN)
		{
			return values.mean();
		}
	else if constexpr(Kp == vext::Op::MIN)
		{
			return values.minCoeff();
		}
	else if constexpr(Kp == vext::Op::MAX)
		{
			return values.maxCoeff();
		}
	else if constexpr(Kp == vext::Op::PROD)
		{
			return values.prod();
		}
	else if constexpr(Kp == vext::Op::L2_NORM)
		{
			return std::sqrt(values.square().sum());
		}
	else
		{
			const float variance = (values - values.mean()).square().mean();
			return Kp == vext::Op::STD ? std::sqrt(variance) : variance;
		}
}

template <vext::Op Kp>
void
BM_VextCpuReduction(
	benchmark::State& state)
{
	const std::uint32_t size = static_cast<std::uint32_t>(state.range(0));

	vext::Tensor<float> values(size);
	vext::Tensor<float> out(1);

	values.set_from(make_values(size, Kp == vext::Op::PROD));

	for([[maybe_unused]] auto iteration : state)
		{
			vext::reduction<Kp>(values, vext::core::no_value_t{}, out);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <vext::Op Kp>
void
BM_EigenCpuReduction(
	benchmark::State& state)
{
	const Eigen::Index size  = state.range(0);
	const auto         input = make_values(size, Kp == vext::Op::PROD);

	const Eigen::Map<const Eigen::ArrayXf> values(input.data(), size);

	float out = 0.0f;

	for([[maybe_unused]] auto iteration : state)
		{
			out = eigen_reduce<Kp>(values);
			benchmark::DoNotOptimize(out);
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <vext::Op Kp>
void
BM_VextCpuAxisReduction(
	benchmark::State& state)
{
	const std::uint32_t rows = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t cols = static_cast<std::uint32_t>(state.range(1));

	vext::Tensor<float> values(rows, cols);
	vext::Tensor<float> out(rows);

	values.set_from(make_values(static_cast<std::int64_t>(rows) * cols, Kp == vext::Op::PROD));

	for([[maybe_unused]] auto iteration : state)
		{
			vext::reduction<Kp>(values, vext::axes({ 1 }), out);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * rows * cols);
}

template <vext::Op Kp>
void
BM_EigenCpuAxisReduction(
	benchmark::State& state)
{
	const Eigen::Index rows  = state.range(0);
	const Eigen::Index cols  = state.range(1);
	const auto         input = make_values(rows * cols, Kp == vext::Op::PROD);

	const Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> values(input.data(), rows, cols);

	Eigen::ArrayXf out(rows);

	for([[maybe_unused]] auto iteration : state)
		{
			for(Eigen::Index row = 0; row < rows; ++row)
				out[row] = eigen_reduce<Kp>(values.row(row).array());

			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * rows * cols);
}

struct CsrData
{
	std::vector<float>         values;
	std::vector<std::uint32_t> head;
	std::vector<std::uint32_t> tail;
};

CsrData
make_csr_data()
{
	CsrData data;
	data.values = make_values(CSR_ROWS * CSR_DEGREE, true);
	data.head.resize(CSR_ROWS + 1);
	data.tail.resize(CSR_ROWS * CSR_DEGREE);

	for(std::int64_t row = 0; row < CSR_ROWS; ++row)
		{
			data.head[row] = static_cast<std::uint32_t>(row * CSR_DEGREE);

			for(std::int64_t entry = 0; entry < CSR_DEGREE; ++entry)
				{
					data.tail[row * CSR_DEGREE + entry] = static_cast<std::uint32_t>((row + entry * 17) % CSR_ROWS);
				}
		}

	data.head.back() = static_cast<std::uint32_t>(CSR_ROWS * CSR_DEGREE);
	return data;
}

template <vext::Op Kp>
Eigen::SparseMatrix<float, Eigen::RowMajor>
make_eigen_sparse(
	const CsrData& data,
	const bool     use_values)
{
	std::vector<Eigen::Triplet<float>> entries;
	entries.reserve(data.tail.size());

	for(std::int64_t row = 0; row < CSR_ROWS; ++row)
		{
			for(std::int64_t entry = 0; entry < CSR_DEGREE; ++entry)
				{
					const auto index = row * CSR_DEGREE + entry;
					float      value = use_values ? data.values[index] : 1.0f;

					if constexpr(Kp == vext::Op::MEAN)
						{
							value /= static_cast<float>(CSR_DEGREE);
						}

					entries.emplace_back(row, data.tail[index], value);
				}
		}

	Eigen::SparseMatrix<float, Eigen::RowMajor> matrix(CSR_ROWS, CSR_ROWS);
	matrix.setFromTriplets(entries.begin(), entries.end());
	return matrix;
}

template <vext::Op Kp>
void
BM_VextCpuCsrScatter(
	benchmark::State& state)
{
	const CsrData data       = make_csr_data();
	const auto    src_values = make_values(CSR_ROWS * CSR_FEATURES);
	const auto    zeros      = std::vector<float>(CSR_ROWS * CSR_FEATURES, 0.0f);

	vext::Tensor<float>         src(CSR_ROWS, CSR_FEATURES);
	vext::Tensor<std::uint32_t> head(CSR_ROWS + 1);
	vext::Tensor<std::uint32_t> tail(CSR_ROWS * CSR_DEGREE);
	vext::Tensor<float>         out(CSR_ROWS, CSR_FEATURES);
	src.set_from(src_values);
	head.set_from(data.head);
	tail.set_from(data.tail);

	for([[maybe_unused]] auto iteration : state)
		{
			state.PauseTiming();
			out.set_from(zeros);
			state.ResumeTiming();
			vext::csr_scatter<Kp>(src, head, tail, out);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}
}

template <vext::Op Kp>
void
BM_EigenCpuCsrScatter(
	benchmark::State& state)
{
	const CsrData data       = make_csr_data();
	const auto    src_values = make_values(CSR_ROWS * CSR_FEATURES);
	const auto    matrix     = make_eigen_sparse<Kp>(data, false);

	const Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> src(src_values.data(), CSR_ROWS, CSR_FEATURES);

	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out(CSR_ROWS, CSR_FEATURES);

	for([[maybe_unused]] auto iteration : state)
		{
			out.noalias() = matrix * src;
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}
}

template <vext::Op Kp>
void
BM_VextCpuCsrSpmv(
	benchmark::State& state)
{
	const CsrData data     = make_csr_data();
	const auto    x_values = make_values(CSR_ROWS);

	vext::Tensor<float>         values(CSR_ROWS * CSR_DEGREE);
	vext::Tensor<std::uint32_t> head(CSR_ROWS + 1);
	vext::Tensor<std::uint32_t> tail(CSR_ROWS * CSR_DEGREE);
	vext::Tensor<float>         x(CSR_ROWS);
	vext::Tensor<float>         out(CSR_ROWS);
	values.set_from(data.values);
	head.set_from(data.head);
	tail.set_from(data.tail);
	x.set_from(x_values);

	for([[maybe_unused]] auto iteration : state)
		{
			vext::csr_spmv<Kp>(values, head, tail, x, out);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}
}

template <vext::Op Kp>
void
BM_EigenCpuCsrSpmv(
	benchmark::State& state)
{
	const CsrData data     = make_csr_data();
	const auto    x_values = make_values(CSR_ROWS);
	const auto    matrix   = make_eigen_sparse<Kp>(data, true);

	const Eigen::Map<const Eigen::VectorXf> x(x_values.data(), CSR_ROWS);

	Eigen::VectorXf out(CSR_ROWS);

	for([[maybe_unused]] auto iteration : state)
		{
			out.noalias() = matrix * x;
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}
}

void
BM_VextCpuMatmul(
	benchmark::State& state)
{
	const std::uint32_t rows   = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t shared = static_cast<std::uint32_t>(state.range(1));
	const std::uint32_t cols   = static_cast<std::uint32_t>(state.range(2));
	const auto          zeros  = std::vector<float>(static_cast<std::size_t>(rows) * cols, 0.0f);

	vext::Tensor<float> lhs(rows, shared);
	vext::Tensor<float> rhs(shared, cols);
	vext::Tensor<float> out(rows, cols);
	lhs.set_from(make_values(static_cast<std::int64_t>(rows) * shared));
	rhs.set_from(make_values(static_cast<std::int64_t>(shared) * cols));

	for([[maybe_unused]] auto iteration : state)
		{
			state.PauseTiming();
			out.set_from(zeros);
			state.ResumeTiming();
			vext::matmul(lhs, rhs, out);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.counters["FLOP/s"] = benchmark::Counter(static_cast<double>(state.iterations()) * 2.0 * rows * shared * cols, benchmark::Counter::kIsRate);
}

void
BM_EigenCpuMatmul(
	benchmark::State& state)
{
	using Matrix                  = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	const Eigen::Index rows       = state.range(0);
	const Eigen::Index shared     = state.range(1);
	const Eigen::Index cols       = state.range(2);
	const auto         lhs_values = make_values(rows * shared);
	const auto         rhs_values = make_values(shared * cols);

	const Eigen::Map<const Matrix> lhs(lhs_values.data(), rows, shared);
	const Eigen::Map<const Matrix> rhs(rhs_values.data(), shared, cols);

	Matrix out(rows, cols);

	for([[maybe_unused]] auto iteration : state)
		{
			out.noalias() = lhs * rhs;
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.counters["FLOP/s"] = benchmark::Counter(static_cast<double>(state.iterations()) * 2.0 * rows * shared * cols, benchmark::Counter::kIsRate);
}

}

#define REGISTER_CPU_PAIR(FAMILY, OP, SIZE)                                                            \
	BENCHMARK_TEMPLATE(BM_VextCpu##FAMILY, vext::Op::OP)->Name("Vext/CPU/" #FAMILY "_" #OP)->Arg(SIZE); \
	BENCHMARK_TEMPLATE(BM_EigenCpu##FAMILY, vext::Op::OP)->Name("Eigen/CPU/" #FAMILY "_" #OP)->Arg(SIZE)

#define REGISTER_CPU_MATRIX_PAIR(FAMILY, OP)                                                                                    \
	BENCHMARK_TEMPLATE(BM_VextCpu##FAMILY, vext::Op::OP)->Name("Vext/CPU/" #FAMILY "_" #OP)->Args({ MATRIX_ROWS, MATRIX_COLS }); \
	BENCHMARK_TEMPLATE(BM_EigenCpu##FAMILY, vext::Op::OP)->Name("Eigen/CPU/" #FAMILY "_" #OP)->Args({ MATRIX_ROWS, MATRIX_COLS })

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

REGISTER_CPU_MATRIX_PAIR(AxisReduction, SUM);
REGISTER_CPU_MATRIX_PAIR(AxisReduction, MEAN);
REGISTER_CPU_MATRIX_PAIR(AxisReduction, MIN);
REGISTER_CPU_MATRIX_PAIR(AxisReduction, MAX);
REGISTER_CPU_MATRIX_PAIR(AxisReduction, PROD);
REGISTER_CPU_MATRIX_PAIR(AxisReduction, STD);
REGISTER_CPU_MATRIX_PAIR(AxisReduction, VAR);
REGISTER_CPU_MATRIX_PAIR(AxisReduction, L2_NORM);

BENCHMARK_TEMPLATE(BM_VextCpuCsrScatter, vext::Op::SUM)->Name("Vext/CPU/CSRScatter_SUM")->Arg(CSR_ROWS);
BENCHMARK_TEMPLATE(BM_EigenCpuCsrScatter, vext::Op::SUM)->Name("Eigen/CPU/CSRScatter_SUM")->Arg(CSR_ROWS);
BENCHMARK_TEMPLATE(BM_VextCpuCsrScatter, vext::Op::MEAN)->Name("Vext/CPU/CSRScatter_MEAN")->Arg(CSR_ROWS);
BENCHMARK_TEMPLATE(BM_EigenCpuCsrScatter, vext::Op::MEAN)->Name("Eigen/CPU/CSRScatter_MEAN")->Arg(CSR_ROWS);
BENCHMARK_TEMPLATE(BM_VextCpuCsrSpmv, vext::Op::SUM)->Name("Vext/CPU/CSRSpMV_SUM")->Arg(CSR_ROWS);
BENCHMARK_TEMPLATE(BM_EigenCpuCsrSpmv, vext::Op::SUM)->Name("Eigen/CPU/CSRSpMV_SUM")->Arg(CSR_ROWS);
BENCHMARK_TEMPLATE(BM_VextCpuCsrSpmv, vext::Op::MEAN)->Name("Vext/CPU/CSRSpMV_MEAN")->Arg(CSR_ROWS);
BENCHMARK_TEMPLATE(BM_EigenCpuCsrSpmv, vext::Op::MEAN)->Name("Eigen/CPU/CSRSpMV_MEAN")->Arg(CSR_ROWS);

BENCHMARK(BM_VextCpuMatmul)->Name("Vext/CPU/Matmul_Square")->Args({ 512, 512, 512 });
BENCHMARK(BM_EigenCpuMatmul)->Name("Eigen/CPU/Matmul_Square")->Args({ 512, 512, 512 });
BENCHMARK(BM_VextCpuMatmul)->Name("Vext/CPU/Matmul_Rectangular")->Args({ 256, 512, 128 });
BENCHMARK(BM_EigenCpuMatmul)->Name("Eigen/CPU/Matmul_Rectangular")->Args({ 256, 512, 128 });

#undef REGISTER_CPU_MATRIX_PAIR
#undef REGISTER_CPU_PAIR

BENCHMARK_MAIN();
