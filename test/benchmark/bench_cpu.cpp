#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <vext/core/cpu/allocator.hpp>
#include <vext/core/cpu/operations/csr_scatter.hpp>
#include <vext/core/cpu/operations/csr_spmv.hpp>
#include <vext/core/cpu/operations/elementwise_binary.hpp>
#include <vext/core/cpu/operations/elementwise_logical.hpp>
#include <vext/core/cpu/operations/elementwise_unary.hpp>
#include <vext/core/cpu/operations/linear_algebra.hpp>
#include <vext/core/cpu/operations/reduction.hpp>
#include <vext/nn/layer/linear.hpp>
#include <vext/tensor.hpp>

namespace
{

constexpr std::uint32_t ELEMENT_COUNT         = 1U << 24U;
constexpr std::uint32_t INPLACE_ELEMENT_COUNT = 1U << 20U;
constexpr std::uint32_t MATRIX_SIZE           = 512U;
constexpr std::uint32_t CSR_ROWS              = 1U << 20U;
constexpr std::uint32_t CSR_FEATURES          = 64U;
constexpr std::uint32_t CSR_DEGREE            = 32U;
constexpr std::int32_t  ELEMENT_ITERS         = 16;
constexpr std::int32_t  MATMUL_ITERS          = 16;

#if defined(__GNUC__) || defined(__clang__)
#define VEXT_BENCHMARK_NOINLINE __attribute__((noinline))
#else
#define VEXT_BENCHMARK_NOINLINE
#endif

VEXT_BENCHMARK_NOINLINE void
observe_buffer(
	std::uint8_t* ptr)
{
	ptr[0] = static_cast<std::uint8_t>(ptr[0] + 1);
	benchmark::DoNotOptimize(ptr[0]);
	benchmark::ClobberMemory();
}

template <typename Tp>
VEXT_BENCHMARK_NOINLINE void
observe_tensor(
	const vext::Tensor<Tp>& tensor)
{
	benchmark::DoNotOptimize(tensor.item(0));
}

std::vector<float>
make_lhs_values(
	const std::uint32_t size)
{
	std::vector<float> values(size, 1.25f);
	return values;
}

std::vector<float>
make_rhs_values(
	const std::uint32_t size)
{
	std::vector<float> values(size, 2.0f);
	return values;
}

template <vext::core::BinaryOperation Kp>
void
BM_CpuBinaryKernel(
	benchmark::State& state)
{
	const std::uint32_t size = static_cast<std::uint32_t>(state.range(0));
	std::vector<float>  lhs  = make_lhs_values(size);
	std::vector<float>  rhs  = make_rhs_values(size);
	std::vector<float>  out(size, 0.0f);

	for([[maybe_unused]] auto iteration : state)
		{
			vext::core::cpu::operations::binary<Kp>(out.data(), lhs.data(), rhs.data(), size);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * sizeof(float) * 3));
}

template <vext::core::BinaryOperation Kp>
void
BM_CpuBinaryTensor(
	benchmark::State& state)
{
	{
		const std::uint32_t rows = static_cast<std::uint32_t>(state.range(0));
		const std::uint32_t cols = static_cast<std::uint32_t>(state.range(1));
		const std::uint32_t size = rows * cols;

		vext::Tensor<float> lhs(rows, cols);
		vext::Tensor<float> rhs(rows, cols);

		lhs.set_from(make_lhs_values(size));
		rhs.set_from(make_rhs_values(size));

		for([[maybe_unused]] auto iteration : state)
			{
				if constexpr(Kp == vext::core::BinaryOperation::ADD)
					{
						const vext::Tensor<float> out = lhs + rhs;
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::BinaryOperation::SUB)
					{
						const vext::Tensor<float> out = lhs - rhs;
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::BinaryOperation::MUL)
					{
						const vext::Tensor<float> out = lhs * rhs;
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::BinaryOperation::DIV)
					{
						const vext::Tensor<float> out = lhs / rhs;
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::BinaryOperation::POW)
					{
						const vext::Tensor<float> out = lhs ^ rhs;
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::BinaryOperation::PRELU)
					{
						vext::Tensor<float> out(lhs);
						out.prelu(rhs);
						observe_tensor(out);
					}
			}

		state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
		state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * sizeof(float) * 3));
	}

	vext::core::cpu::allocator::free();
}

template <vext::core::BinaryOperation Kp>
void
BM_CpuBinaryTensorInPlace(
	benchmark::State& state)
{
	{
		const std::uint32_t rows = static_cast<std::uint32_t>(state.range(0));
		const std::uint32_t cols = static_cast<std::uint32_t>(state.range(1));
		const std::uint32_t size = rows * cols;

		vext::Tensor<float> source(rows, cols);
		vext::Tensor<float> rhs(rows, cols);

		source.set_from(make_lhs_values(size));
		rhs.set_from(make_rhs_values(size));

		vext::Tensor<float> out(source);

		for([[maybe_unused]] auto iteration : state)
			{
				if constexpr(Kp == vext::core::BinaryOperation::ADD)
					{
						out += rhs;
					}
				else if constexpr(Kp == vext::core::BinaryOperation::SUB)
					{
						out -= rhs;
					}
				else if constexpr(Kp == vext::core::BinaryOperation::MUL)
					{
						out *= rhs;
					}
				else if constexpr(Kp == vext::core::BinaryOperation::DIV)
					{
						out /= rhs;
					}
				else if constexpr(Kp == vext::core::BinaryOperation::POW)
					{
						out ^= rhs;
					}

				observe_tensor(out);
			}

		state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
		state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * sizeof(float) * 3));
	}

	vext::core::cpu::allocator::free();
}

template <vext::core::LogicOperation Kp>
void
BM_CpuLogicalKernel(
	benchmark::State& state)
{
	const std::uint32_t size = static_cast<std::uint32_t>(state.range(0));

	std::vector<float>        lhs = make_lhs_values(size);
	std::vector<float>        rhs = make_rhs_values(size);
	std::vector<std::uint8_t> out(size, 0);

	for([[maybe_unused]] auto iteration : state)
		{
			vext::core::cpu::operations::logical<Kp>(out.data(), lhs.data(), rhs.data(), size);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * (sizeof(float) * 2 + sizeof(std::uint8_t))));
}

template <vext::core::LogicOperation Kp>
void
BM_CpuLogicalTensor(
	benchmark::State& state)
{
	{
		const std::uint32_t rows = static_cast<std::uint32_t>(state.range(0));
		const std::uint32_t cols = static_cast<std::uint32_t>(state.range(1));
		const std::uint32_t size = rows * cols;

		vext::Tensor<float> lhs(rows, cols);
		vext::Tensor<float> rhs(rows, cols);

		lhs.set_from(make_lhs_values(size));
		rhs.set_from(make_rhs_values(size));

		for([[maybe_unused]] auto iteration : state)
			{
				if constexpr(Kp == vext::core::LogicOperation::EQUAL)
					{
						const vext::Tensor<std::uint8_t> out = lhs == rhs;
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::LogicOperation::NOT_EQUAL)
					{
						const vext::Tensor<std::uint8_t> out = lhs != rhs;
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::LogicOperation::LESS)
					{
						const vext::Tensor<std::uint8_t> out = lhs < rhs;
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::LogicOperation::LESS_EQUAL)
					{
						const vext::Tensor<std::uint8_t> out = lhs <= rhs;
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::LogicOperation::GREATER)
					{
						const vext::Tensor<std::uint8_t> out = lhs > rhs;
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::LogicOperation::GREATER_EQUAL)
					{
						const vext::Tensor<std::uint8_t> out = lhs >= rhs;
						observe_tensor(out);
					}
			}

		state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
		state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * (sizeof(float) * 2 + sizeof(std::uint8_t))));
	}

	vext::core::cpu::allocator::free();
}

template <vext::core::UnaryOperation Kp>
void
BM_CpuUnaryKernel(
	benchmark::State& state)
{
	const std::uint32_t size = static_cast<std::uint32_t>(state.range(0));

	std::vector<float> values(size, 0.5f);

	for([[maybe_unused]] auto iteration : state)
		{
			if constexpr(Kp == vext::core::UnaryOperation::LEAKY_RELU || Kp == vext::core::UnaryOperation::ELU || Kp == vext::core::UnaryOperation::SWISH)
				{
					vext::core::cpu::operations::unary<Kp>(values.data(), size, 0.25f);
				}
			else if constexpr(Kp == vext::core::UnaryOperation::LINEAR || Kp == vext::core::UnaryOperation::CLIP || Kp == vext::core::UnaryOperation::POW)
				{
					vext::core::cpu::operations::unary<Kp>(values.data(), size, 0.75f, 1.25f);
				}
			else
				{
					vext::core::cpu::operations::unary<Kp>(values.data(), size);
				}

			benchmark::DoNotOptimize(values.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * sizeof(float)));
}

template <vext::core::UnaryOperation Kp>
void
BM_CpuUnaryTensor(
	benchmark::State& state)
{
	{
		const std::uint32_t rows = static_cast<std::uint32_t>(state.range(0));
		const std::uint32_t cols = static_cast<std::uint32_t>(state.range(1));
		const std::uint32_t size = rows * cols;

		vext::Tensor<float> source(rows, cols);
		source.set_from(std::vector<float>(size, 0.5f));

		for([[maybe_unused]] auto iteration : state)
			{
				vext::Tensor<float> out(source);

				if constexpr(Kp == vext::core::UnaryOperation::ABS)
					{
						out.abs();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::SIN)
					{
						out.sin();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::COS)
					{
						out.cos();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::NEG)
					{
						-out;
					}
				else if constexpr(Kp == vext::core::UnaryOperation::EXP)
					{
						out.exp();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::LOG)
					{
						out.log();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::SQRT)
					{
						out.sqrt();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::SQUARE)
					{
						out.square();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::ROUND)
					{
						out.round();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::SIGMOID)
					{
						out.sigmoid();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::SOFT_RELU)
					{
						out.soft_relu();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::RELU)
					{
						out.relu();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::SOFTMAX)
					{
						out.softmax();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::SOFTMIN)
					{
						out.softmin();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::LOGSOFTMAX)
					{
						out.log_softmax();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::LEAKY_RELU)
					{
						out.leaky_relu(0.25f);
					}
				else if constexpr(Kp == vext::core::UnaryOperation::ELU)
					{
						out.elu(0.25f);
					}
				else if constexpr(Kp == vext::core::UnaryOperation::SWISH)
					{
						out.swish(0.25f);
					}
				else if constexpr(Kp == vext::core::UnaryOperation::LINEAR)
					{
						out.linear(0.75f, 1.25f);
					}
				else if constexpr(Kp == vext::core::UnaryOperation::CLIP)
					{
						out.clip(0.25f, 0.75f);
					}
				else if constexpr(Kp == vext::core::UnaryOperation::POW)
					{
						out.pow(0.75f, 1.25f);
					}

				observe_tensor(out);
			}

		state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
		state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * sizeof(float)));
	}

	vext::core::cpu::allocator::free();
}

template <vext::core::ReductionOperation Kp>
void
BM_CpuReductionKernel(
	benchmark::State& state)
{
	const std::uint32_t size = static_cast<std::uint32_t>(state.range(0));

	std::vector<float> values(size, 1.0f);
	float              result = 0.0f;

	const std::vector<std::uint32_t> keep_dims{ 1 };
	const std::vector<std::uint32_t> keep_strides{ 0 };
	const std::vector<std::uint32_t> reduce_dims{ size };
	const std::vector<std::uint32_t> reduce_strides{ 1 };

	for([[maybe_unused]] auto iteration : state)
		{
			vext::core::cpu::operations::reduce<Kp>(&result, values.data(), 1, size, keep_dims, keep_strides, reduce_dims, reduce_strides);
			benchmark::DoNotOptimize(result);
		}

	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * sizeof(float)));
}

template <vext::core::ReductionOperation Kp>
void
BM_CpuReductionTensor(
	benchmark::State& state)
{
	{
		const std::uint32_t size = static_cast<std::uint32_t>(state.range(0));

		vext::Tensor<float> tensor(size);
		tensor.set_from(std::vector<float>(size, 1.0f));

		for([[maybe_unused]] auto iteration : state)
			{
				if constexpr(Kp == vext::core::ReductionOperation::SUM)
					{
						const vext::Tensor<float> out = tensor.sum();
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::ReductionOperation::MEAN)
					{
						const vext::Tensor<float> out = tensor.mean();
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::ReductionOperation::MAX)
					{
						const vext::Tensor<float> out = tensor.max();
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::ReductionOperation::MIN)
					{
						const vext::Tensor<float> out = tensor.min();
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::ReductionOperation::PROD)
					{
						const vext::Tensor<float> out = tensor.prod();
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::ReductionOperation::STD)
					{
						const vext::Tensor<float> out = tensor.std();
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::ReductionOperation::VAR)
					{
						const vext::Tensor<float> out = tensor.var();
						observe_tensor(out);
					}
			}

		state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
		state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * sizeof(float)));
	}

	vext::core::cpu::allocator::free();
}

std::vector<std::uint32_t>
make_csr_head(
	const std::uint32_t rows,
	const std::uint32_t degree)
{
	std::vector<std::uint32_t> head(rows + 1U, 0U);

	for(std::uint32_t row = 0; row <= rows; ++row)
		{
			head[row] = row * degree;
		}

	return head;
}

std::vector<std::uint32_t>
make_csr_tail(
	const std::uint32_t rows,
	const std::uint32_t degree)
{
	std::vector<std::uint32_t> tail(rows * degree, 0U);

	for(std::uint32_t row = 0; row < rows; ++row)
		{
			for(std::uint32_t edge = 0; edge < degree; ++edge)
				{
					tail[row * degree + edge] = (row + edge) % rows;
				}
		}

	return tail;
}

template <vext::core::CSRScatterOperation Kp>
void
BM_CpuCsrScatterKernel(
	benchmark::State& state)
{
	const std::uint32_t rows     = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t features = static_cast<std::uint32_t>(state.range(1));
	const std::uint32_t degree   = static_cast<std::uint32_t>(state.range(2));
	const std::uint32_t size     = rows * features;

	std::vector<float>         src(size, 1.25f);
	std::vector<float>         out(size, 0.0f);
	std::vector<std::uint32_t> head = make_csr_head(rows, degree);
	std::vector<std::uint32_t> tail = make_csr_tail(rows, degree);

	for([[maybe_unused]] auto iteration : state)
		{
			if constexpr(Kp == vext::core::CSRScatterOperation::MIN)
				{
					std::fill(out.begin(), out.end(), std::numeric_limits<float>::max());
				}
			else if constexpr(Kp == vext::core::CSRScatterOperation::PROD)
				{
					std::fill(out.begin(), out.end(), 1.0f);
				}
			else
				{
					std::fill(out.begin(), out.end(), 0.0f);
				}

			vext::core::cpu::operations::csr_scatter<Kp>(out.data(), src.data(), head.data(), tail.data(), rows, features);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * rows * degree * features));
	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * rows * degree * features * sizeof(float)));
}

template <vext::core::CSRScatterOperation Kp>
void
BM_CpuCsrScatterTensor(
	benchmark::State& state)
{
	{
		const std::uint32_t rows     = static_cast<std::uint32_t>(state.range(0));
		const std::uint32_t features = static_cast<std::uint32_t>(state.range(1));
		const std::uint32_t degree   = static_cast<std::uint32_t>(state.range(2));
		const std::uint32_t size     = rows * features;

		vext::Tensor<float>         src(rows, features);
		vext::Tensor<std::uint32_t> head(rows + 1U);
		vext::Tensor<std::uint32_t> tail(rows * degree);

		src.set_from(std::vector<float>(size, 1.25f));
		head.set_from(make_csr_head(rows, degree));
		tail.set_from(make_csr_tail(rows, degree));

		for([[maybe_unused]] auto iteration : state)
			{
				if constexpr(Kp == vext::core::CSRScatterOperation::SUM)
					{
						const vext::Tensor<float> out = src.csr_scatter_sum(head, tail);
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::CSRScatterOperation::MEAN)
					{
						const vext::Tensor<float> out = src.csr_scatter_mean(head, tail);
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::CSRScatterOperation::MAX)
					{
						const vext::Tensor<float> out = src.csr_scatter_max(head, tail);
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::CSRScatterOperation::MIN)
					{
						const vext::Tensor<float> out = src.csr_scatter_min(head, tail);
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::CSRScatterOperation::PROD)
					{
						const vext::Tensor<float> out = src.csr_scatter_prod(head, tail);
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::CSRScatterOperation::STD)
					{
						const vext::Tensor<float> out = src.csr_scatter_std(head, tail);
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::CSRScatterOperation::VAR)
					{
						const vext::Tensor<float> out = src.csr_scatter_var(head, tail);
						observe_tensor(out);
					}
			}

		state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * rows * degree * features));
		state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * rows * degree * features * sizeof(float)));
	}

	vext::core::cpu::allocator::free();
}

template <vext::core::CSRSpMVOperation Kp>
void
BM_CpuCsrSpmvKernel(
	benchmark::State& state)
{
	const std::uint32_t rows   = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t degree = static_cast<std::uint32_t>(state.range(1));
	const std::uint32_t nnz    = rows * degree;

	std::vector<float>         values(nnz, 1.25f);
	std::vector<float>         x(rows, 2.0f);
	std::vector<float>         out(rows, 0.0f);
	std::vector<std::uint32_t> head = make_csr_head(rows, degree);
	std::vector<std::uint32_t> tail = make_csr_tail(rows, degree);

	for([[maybe_unused]] auto iteration : state)
		{
			vext::core::cpu::operations::csr_spmv<Kp>(out.data(), values.data(), head.data(), tail.data(), x.data(), rows);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * nnz));
	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * nnz * (sizeof(float) * 2 + sizeof(std::uint32_t))));
}

template <vext::core::CSRSpMVOperation Kp>
void
BM_CpuCsrSpmvTensor(
	benchmark::State& state)
{
	{
		const std::uint32_t rows   = static_cast<std::uint32_t>(state.range(0));
		const std::uint32_t degree = static_cast<std::uint32_t>(state.range(1));
		const std::uint32_t nnz    = rows * degree;

		vext::Tensor<float>         values(nnz);
		vext::Tensor<std::uint32_t> head(rows + 1U);
		vext::Tensor<std::uint32_t> tail(nnz);
		vext::Tensor<float>         x(rows);

		values.set_from(std::vector<float>(nnz, 1.25f));
		head.set_from(make_csr_head(rows, degree));
		tail.set_from(make_csr_tail(rows, degree));
		x.set_from(std::vector<float>(rows, 2.0f));

		for([[maybe_unused]] auto iteration : state)
			{
				if constexpr(Kp == vext::core::CSRSpMVOperation::SUM)
					{
						const vext::Tensor<float> out = values.csr_spmv_sum(head, tail, x);
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::CSRSpMVOperation::MEAN)
					{
						const vext::Tensor<float> out = values.csr_spmv_mean(head, tail, x);
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::CSRSpMVOperation::MAX)
					{
						const vext::Tensor<float> out = values.csr_spmv_max(head, tail, x);
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::CSRSpMVOperation::MIN)
					{
						const vext::Tensor<float> out = values.csr_spmv_min(head, tail, x);
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::CSRSpMVOperation::PROD)
					{
						const vext::Tensor<float> out = values.csr_spmv_prod(head, tail, x);
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::CSRSpMVOperation::STD)
					{
						const vext::Tensor<float> out = values.csr_spmv_std(head, tail, x);
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::CSRSpMVOperation::VAR)
					{
						const vext::Tensor<float> out = values.csr_spmv_var(head, tail, x);
						observe_tensor(out);
					}
			}

		state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * nnz));
		state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * nnz * (sizeof(float) * 2 + sizeof(std::uint32_t))));
	}

	vext::core::cpu::allocator::free();
}

void
BM_CpuMatmulKernel(
	benchmark::State& state)
{
	const std::uint32_t m = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t p = static_cast<std::uint32_t>(state.range(1));
	const std::uint32_t n = static_cast<std::uint32_t>(state.range(2));

	std::vector<float> lhs(m * p, 0.5f);
	std::vector<float> rhs(p * n, 0.25f);
	std::vector<float> out(m * n, 0.0f);

	for([[maybe_unused]] auto iteration : state)
		{
			std::fill(out.begin(), out.end(), 0.0f);
			vext::core::cpu::operations::matmul(out.data(), lhs.data(), rhs.data(), m, p, n);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * m * n));
	state.counters["flop"] = benchmark::Counter(static_cast<double>(state.iterations()) * 2.0 * m * n * p, benchmark::Counter::kIsRate);
}

void
BM_CpuTensorMatmul(
	benchmark::State& state)
{
	{
		const std::uint32_t m = static_cast<std::uint32_t>(state.range(0));
		const std::uint32_t p = static_cast<std::uint32_t>(state.range(1));
		const std::uint32_t n = static_cast<std::uint32_t>(state.range(2));

		const vext::Tensor<float> lhs(m, p);
		const vext::Tensor<float> rhs(p, n);

		for([[maybe_unused]] auto iteration : state)
			{
				const vext::Tensor<float> out = lhs.matmul(rhs);
				observe_tensor(out);
			}

		state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * m * n));
		state.counters["flop"] = benchmark::Counter(static_cast<double>(state.iterations()) * 2.0 * m * n * p, benchmark::Counter::kIsRate);
	}

	vext::core::cpu::allocator::free();
}

void
BM_CpuNnLinearForward(
	benchmark::State& state)
{
	{
		const std::uint32_t                               size = static_cast<std::uint32_t>(state.range(0));
		const vext::Tensor<float>                         input(size, size);
		const vext::nn::layer::Linear<vext::Backend::CPU> layer(size, size);

		for([[maybe_unused]] auto iteration : state)
			{
				const vext::Tensor<float> out = layer(input);
				observe_tensor(out);
			}

		state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size * size));
		state.counters["flop"] = benchmark::Counter(static_cast<double>(state.iterations()) * 2.0 * size * size * size, benchmark::Counter::kIsRate);
	}

	vext::core::cpu::allocator::free();
}

void
BM_CpuAllocatorSmall(
	benchmark::State& state)
{
	const std::uint64_t bytes = static_cast<std::uint64_t>(state.range(0));

	for([[maybe_unused]] auto iteration : state)
		{
			std::uint8_t* ptr = vext::core::cpu::allocator::allocate<std::uint8_t>(bytes);
			observe_buffer(ptr);
			vext::core::cpu::allocator::deallocate(ptr);
		}

	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * bytes));

	vext::core::cpu::allocator::free();
}

void
BM_CpuAllocatorLarge(
	benchmark::State& state)
{
	const std::uint64_t bytes = static_cast<std::uint64_t>(state.range(0));

	for([[maybe_unused]] auto iteration : state)
		{
			std::uint8_t* ptr = vext::core::cpu::allocator::allocate<std::uint8_t>(bytes);
			observe_buffer(ptr);
			vext::core::cpu::allocator::deallocate(ptr);
		}

	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * bytes));

	vext::core::cpu::allocator::free();
}

void
BM_CpuTensorConstruct(
	benchmark::State& state)
{
	{
		const std::uint32_t rows = static_cast<std::uint32_t>(state.range(0));
		const std::uint32_t cols = static_cast<std::uint32_t>(state.range(1));

		for([[maybe_unused]] auto iteration : state)
			{
				const vext::Tensor<float> tensor(rows, cols);
				observe_tensor(tensor);
			}

		state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * rows * cols));
	}

	vext::core::cpu::allocator::free();
}

void
BM_CpuTensorCopy(
	benchmark::State& state)
{
	{
		const std::uint32_t       rows = static_cast<std::uint32_t>(state.range(0));
		const std::uint32_t       cols = static_cast<std::uint32_t>(state.range(1));
		const vext::Tensor<float> source(rows, cols);

		for([[maybe_unused]] auto iteration : state)
			{
				const vext::Tensor<float> copy(source);
				observe_tensor(copy);
			}

		state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * rows * cols * sizeof(float)));
	}

	vext::core::cpu::allocator::free();
}

}

BENCHMARK(BM_CpuAllocatorSmall)->Arg(4096)->Iterations(200000);
BENCHMARK(BM_CpuAllocatorLarge)->Arg(32 * 1024 * 1024)->Iterations(128);
BENCHMARK(BM_CpuTensorConstruct)->Args({ 4096, 4096 })->Iterations(32);
BENCHMARK(BM_CpuTensorCopy)->Args({ 4096, 4096 })->Iterations(32);

BENCHMARK_TEMPLATE(BM_CpuBinaryKernel, vext::core::BinaryOperation::ADD)->Name("BM_CpuBinaryKernel/ADD")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuBinaryTensor, vext::core::BinaryOperation::ADD)->Name("BM_CpuBinaryTensor/ADD")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuBinaryTensorInPlace, vext::core::BinaryOperation::ADD)->Name("BM_CpuBinaryTensorInPlace/ADD")->Args({ INPLACE_ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuBinaryKernel, vext::core::BinaryOperation::SUB)->Name("BM_CpuBinaryKernel/SUB")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuBinaryTensor, vext::core::BinaryOperation::SUB)->Name("BM_CpuBinaryTensor/SUB")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuBinaryTensorInPlace, vext::core::BinaryOperation::SUB)->Name("BM_CpuBinaryTensorInPlace/SUB")->Args({ INPLACE_ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuBinaryKernel, vext::core::BinaryOperation::MUL)->Name("BM_CpuBinaryKernel/MUL")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuBinaryTensor, vext::core::BinaryOperation::MUL)->Name("BM_CpuBinaryTensor/MUL")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuBinaryTensorInPlace, vext::core::BinaryOperation::MUL)->Name("BM_CpuBinaryTensorInPlace/MUL")->Args({ INPLACE_ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuBinaryKernel, vext::core::BinaryOperation::DIV)->Name("BM_CpuBinaryKernel/DIV")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuBinaryTensor, vext::core::BinaryOperation::DIV)->Name("BM_CpuBinaryTensor/DIV")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuBinaryTensorInPlace, vext::core::BinaryOperation::DIV)->Name("BM_CpuBinaryTensorInPlace/DIV")->Args({ INPLACE_ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuBinaryKernel, vext::core::BinaryOperation::POW)->Name("BM_CpuBinaryKernel/POW")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuBinaryTensor, vext::core::BinaryOperation::POW)->Name("BM_CpuBinaryTensor/POW")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuBinaryTensorInPlace, vext::core::BinaryOperation::POW)->Name("BM_CpuBinaryTensorInPlace/POW")->Args({ INPLACE_ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuBinaryKernel, vext::core::BinaryOperation::PRELU)->Name("BM_CpuBinaryKernel/PRELU")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuBinaryTensor, vext::core::BinaryOperation::PRELU)->Name("BM_CpuBinaryTensor/PRELU")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuBinaryKernel, vext::core::BinaryOperation::MIN)->Name("BM_CpuBinaryKernel/MIN/kernel_only")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuBinaryKernel, vext::core::BinaryOperation::MAX)->Name("BM_CpuBinaryKernel/MAX/kernel_only")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);

BENCHMARK_TEMPLATE(BM_CpuLogicalKernel, vext::core::LogicOperation::EQUAL)->Name("BM_CpuLogicalKernel/EQUAL")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuLogicalTensor, vext::core::LogicOperation::EQUAL)->Name("BM_CpuLogicalTensor/EQUAL")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuLogicalKernel, vext::core::LogicOperation::NOT_EQUAL)->Name("BM_CpuLogicalKernel/NOT_EQUAL")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuLogicalTensor, vext::core::LogicOperation::NOT_EQUAL)->Name("BM_CpuLogicalTensor/NOT_EQUAL")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuLogicalKernel, vext::core::LogicOperation::LESS)->Name("BM_CpuLogicalKernel/LESS")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuLogicalTensor, vext::core::LogicOperation::LESS)->Name("BM_CpuLogicalTensor/LESS")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuLogicalKernel, vext::core::LogicOperation::LESS_EQUAL)->Name("BM_CpuLogicalKernel/LESS_EQUAL")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuLogicalTensor, vext::core::LogicOperation::LESS_EQUAL)->Name("BM_CpuLogicalTensor/LESS_EQUAL")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuLogicalKernel, vext::core::LogicOperation::GREATER)->Name("BM_CpuLogicalKernel/GREATER")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuLogicalTensor, vext::core::LogicOperation::GREATER)->Name("BM_CpuLogicalTensor/GREATER")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuLogicalKernel, vext::core::LogicOperation::GREATER_EQUAL)->Name("BM_CpuLogicalKernel/GREATER_EQUAL")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuLogicalTensor, vext::core::LogicOperation::GREATER_EQUAL)->Name("BM_CpuLogicalTensor/GREATER_EQUAL")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);

BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::ABS)->Name("BM_CpuUnaryKernel/ABS")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::ABS)->Name("BM_CpuUnaryTensor/ABS")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::SIN)->Name("BM_CpuUnaryKernel/SIN")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::SIN)->Name("BM_CpuUnaryTensor/SIN")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::COS)->Name("BM_CpuUnaryKernel/COS")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::COS)->Name("BM_CpuUnaryTensor/COS")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::TANH)->Name("BM_CpuUnaryKernel/TANH/kernel_only")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::NEG)->Name("BM_CpuUnaryKernel/NEG")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::NEG)->Name("BM_CpuUnaryTensor/NEG")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::EXP)->Name("BM_CpuUnaryKernel/EXP")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::EXP)->Name("BM_CpuUnaryTensor/EXP")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::LOG)->Name("BM_CpuUnaryKernel/LOG")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::LOG)->Name("BM_CpuUnaryTensor/LOG")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::SQRT)->Name("BM_CpuUnaryKernel/SQRT")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::SQRT)->Name("BM_CpuUnaryTensor/SQRT")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::SQUARE)->Name("BM_CpuUnaryKernel/SQUARE")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::SQUARE)->Name("BM_CpuUnaryTensor/SQUARE")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::ROUND)->Name("BM_CpuUnaryKernel/ROUND")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::ROUND)->Name("BM_CpuUnaryTensor/ROUND")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::SIGMOID)->Name("BM_CpuUnaryKernel/SIGMOID")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::SIGMOID)->Name("BM_CpuUnaryTensor/SIGMOID")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::SOFT_RELU)->Name("BM_CpuUnaryKernel/SOFT_RELU")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::SOFT_RELU)->Name("BM_CpuUnaryTensor/SOFT_RELU")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::RELU)->Name("BM_CpuUnaryKernel/RELU")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::RELU)->Name("BM_CpuUnaryTensor/RELU")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::SOFTMAX)->Name("BM_CpuUnaryKernel/SOFTMAX")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::SOFTMAX)->Name("BM_CpuUnaryTensor/SOFTMAX")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::SOFTMIN)->Name("BM_CpuUnaryKernel/SOFTMIN")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::SOFTMIN)->Name("BM_CpuUnaryTensor/SOFTMIN")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::LOGSOFTMAX)->Name("BM_CpuUnaryKernel/LOGSOFTMAX")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::LOGSOFTMAX)->Name("BM_CpuUnaryTensor/LOGSOFTMAX")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::LEAKY_RELU)->Name("BM_CpuUnaryKernel/LEAKY_RELU")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::LEAKY_RELU)->Name("BM_CpuUnaryTensor/LEAKY_RELU")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::ELU)->Name("BM_CpuUnaryKernel/ELU")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::ELU)->Name("BM_CpuUnaryTensor/ELU")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::SWISH)->Name("BM_CpuUnaryKernel/SWISH")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::SWISH)->Name("BM_CpuUnaryTensor/SWISH")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::LINEAR)->Name("BM_CpuUnaryKernel/LINEAR")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::LINEAR)->Name("BM_CpuUnaryTensor/LINEAR")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::CLIP)->Name("BM_CpuUnaryKernel/CLIP")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::CLIP)->Name("BM_CpuUnaryTensor/CLIP")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryKernel, vext::core::UnaryOperation::POW)->Name("BM_CpuUnaryKernel/UNARY_POW")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuUnaryTensor, vext::core::UnaryOperation::POW)->Name("BM_CpuUnaryTensor/UNARY_POW")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS);

BENCHMARK_TEMPLATE(BM_CpuReductionKernel, vext::core::ReductionOperation::SUM)->Name("BM_CpuReductionKernel/SUM")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuReductionTensor, vext::core::ReductionOperation::SUM)->Name("BM_CpuReductionTensor/SUM")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuReductionKernel, vext::core::ReductionOperation::MEAN)->Name("BM_CpuReductionKernel/MEAN")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuReductionTensor, vext::core::ReductionOperation::MEAN)->Name("BM_CpuReductionTensor/MEAN")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuReductionKernel, vext::core::ReductionOperation::MAX)->Name("BM_CpuReductionKernel/MAX")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuReductionTensor, vext::core::ReductionOperation::MAX)->Name("BM_CpuReductionTensor/MAX")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuReductionKernel, vext::core::ReductionOperation::MIN)->Name("BM_CpuReductionKernel/MIN")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuReductionTensor, vext::core::ReductionOperation::MIN)->Name("BM_CpuReductionTensor/MIN")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuReductionKernel, vext::core::ReductionOperation::PROD)->Name("BM_CpuReductionKernel/PROD")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuReductionTensor, vext::core::ReductionOperation::PROD)->Name("BM_CpuReductionTensor/PROD")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuReductionKernel, vext::core::ReductionOperation::STD)->Name("BM_CpuReductionKernel/STD")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuReductionTensor, vext::core::ReductionOperation::STD)->Name("BM_CpuReductionTensor/STD")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuReductionKernel, vext::core::ReductionOperation::VAR)->Name("BM_CpuReductionKernel/VAR")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuReductionTensor, vext::core::ReductionOperation::VAR)->Name("BM_CpuReductionTensor/VAR")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS);

BENCHMARK_TEMPLATE(BM_CpuCsrScatterKernel, vext::core::CSRScatterOperation::SUM)->Name("BM_CpuCsrScatterKernel/SUM")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrScatterTensor, vext::core::CSRScatterOperation::SUM)->Name("BM_CpuCsrScatterTensor/SUM")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrScatterKernel, vext::core::CSRScatterOperation::MEAN)->Name("BM_CpuCsrScatterKernel/MEAN")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrScatterTensor, vext::core::CSRScatterOperation::MEAN)->Name("BM_CpuCsrScatterTensor/MEAN")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrScatterKernel, vext::core::CSRScatterOperation::MAX)->Name("BM_CpuCsrScatterKernel/MAX")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrScatterTensor, vext::core::CSRScatterOperation::MAX)->Name("BM_CpuCsrScatterTensor/MAX")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrScatterKernel, vext::core::CSRScatterOperation::MIN)->Name("BM_CpuCsrScatterKernel/MIN")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrScatterTensor, vext::core::CSRScatterOperation::MIN)->Name("BM_CpuCsrScatterTensor/MIN")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrScatterKernel, vext::core::CSRScatterOperation::PROD)->Name("BM_CpuCsrScatterKernel/PROD")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrScatterTensor, vext::core::CSRScatterOperation::PROD)->Name("BM_CpuCsrScatterTensor/PROD")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrScatterKernel, vext::core::CSRScatterOperation::STD)->Name("BM_CpuCsrScatterKernel/STD")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrScatterTensor, vext::core::CSRScatterOperation::STD)->Name("BM_CpuCsrScatterTensor/STD")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrScatterKernel, vext::core::CSRScatterOperation::VAR)->Name("BM_CpuCsrScatterKernel/VAR")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrScatterTensor, vext::core::CSRScatterOperation::VAR)->Name("BM_CpuCsrScatterTensor/VAR")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS);

BENCHMARK_TEMPLATE(BM_CpuCsrSpmvKernel, vext::core::CSRSpMVOperation::SUM)->Name("BM_CpuCsrSpmvKernel/SUM")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrSpmvTensor, vext::core::CSRSpMVOperation::SUM)->Name("BM_CpuCsrSpmvTensor/SUM")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrSpmvKernel, vext::core::CSRSpMVOperation::MEAN)->Name("BM_CpuCsrSpmvKernel/MEAN")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrSpmvTensor, vext::core::CSRSpMVOperation::MEAN)->Name("BM_CpuCsrSpmvTensor/MEAN")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrSpmvKernel, vext::core::CSRSpMVOperation::MAX)->Name("BM_CpuCsrSpmvKernel/MAX")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrSpmvTensor, vext::core::CSRSpMVOperation::MAX)->Name("BM_CpuCsrSpmvTensor/MAX")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrSpmvKernel, vext::core::CSRSpMVOperation::MIN)->Name("BM_CpuCsrSpmvKernel/MIN")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrSpmvTensor, vext::core::CSRSpMVOperation::MIN)->Name("BM_CpuCsrSpmvTensor/MIN")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrSpmvKernel, vext::core::CSRSpMVOperation::PROD)->Name("BM_CpuCsrSpmvKernel/PROD")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrSpmvTensor, vext::core::CSRSpMVOperation::PROD)->Name("BM_CpuCsrSpmvTensor/PROD")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrSpmvKernel, vext::core::CSRSpMVOperation::STD)->Name("BM_CpuCsrSpmvKernel/STD")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrSpmvTensor, vext::core::CSRSpMVOperation::STD)->Name("BM_CpuCsrSpmvTensor/STD")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrSpmvKernel, vext::core::CSRSpMVOperation::VAR)->Name("BM_CpuCsrSpmvKernel/VAR")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS);
BENCHMARK_TEMPLATE(BM_CpuCsrSpmvTensor, vext::core::CSRSpMVOperation::VAR)->Name("BM_CpuCsrSpmvTensor/VAR")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS);

BENCHMARK(BM_CpuMatmulKernel)->Args({ MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE })->Iterations(MATMUL_ITERS);
BENCHMARK(BM_CpuTensorMatmul)->Args({ MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE })->Iterations(MATMUL_ITERS);
BENCHMARK(BM_CpuNnLinearForward)->Arg(MATRIX_SIZE)->Iterations(MATMUL_ITERS);

BENCHMARK_MAIN();

#undef VEXT_BENCHMARK_NOINLINE
