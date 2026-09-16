#ifndef __VEXT_BENCHMARKS_CPU_ELEMENTWISE_BINARY_HPP__
#define __VEXT_BENCHMARKS_CPU_ELEMENTWISE_BINARY_HPP__

#include <benchmark/benchmark.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <vext/ops.hpp>
#include <vext/tensor.hpp>

#include <cpu/common.hpp>

namespace vext::benchmarks::cpu::ops
{

template <::vext::Op Kp, typename Out, typename Lhs, typename Rhs>
void
run_eigen_binary(
	Out&       out,
	const Lhs& lhs,
	const Rhs& rhs)
{
	if constexpr(Kp == ::vext::Op::ADD)
		{
			out = lhs + rhs;
		}
	else if constexpr(Kp == ::vext::Op::SUB)
		{
			out = lhs - rhs;
		}
	else if constexpr(Kp == ::vext::Op::MUL)
		{
			out = lhs * rhs;
		}
	else if constexpr(Kp == ::vext::Op::DIV)
		{
			out = lhs / rhs;
		}
	else if constexpr(Kp == ::vext::Op::POW)
		{
			out = lhs.pow(rhs);
		}
	else if constexpr(Kp == ::vext::Op::MIN)
		{
			out = lhs.min(rhs);
		}
	else if constexpr(Kp == ::vext::Op::MAX)
		{
			out = lhs.max(rhs);
		}
	else
		{
			out = lhs.max(0.0f) + rhs * lhs.min(0.0f);
		}
}

template <::vext::Op Kp>
void
BM_VextCpuBinary(
	benchmark::State& state)
{
	const std::vector<std::uint32_t> shape = shape_from_state(state);
	const std::uint32_t              size  = shape_length(shape);

	::vext::Tensor<float> lhs(shape);
	::vext::Tensor<float> rhs(shape);
	::vext::Tensor<float> out(shape);

	lhs.set_from(make_values(size, Kp == ::vext::Op::POW));
	rhs.set_from(make_values(size, true));

	for([[maybe_unused]] auto iteration : state)
		{
			::vext::binary<Kp>(lhs, rhs, out);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <::vext::Op Kp>
void
BM_EigenCpuBinary(
	benchmark::State& state)
{
	const std::vector<std::uint32_t>       shape      = shape_from_state(state);
	const Eigen::Index                     size       = shape_length(shape);
	const std::vector<float>               lhs_values = make_values(size, Kp == ::vext::Op::POW);
	const std::vector<float>               rhs_values = make_values(size, true);
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

template <::vext::Op Kp>
void
BM_VextCpuBroadcast(
	benchmark::State& state)
{
	const std::vector<std::uint32_t> shape = shape_from_state(state);
	const std::uint32_t              size  = shape_length(shape);
	const std::uint32_t              cols  = shape.back();

	::vext::Tensor<float> matrix(shape);
	::vext::Tensor<float> row(cols);
	::vext::Tensor<float> out(shape);

	matrix.set_from(make_values(size, Kp == ::vext::Op::POW));
	row.set_from(make_values(cols, true));

	for([[maybe_unused]] auto iteration : state)
		{
			::vext::binary<Kp>(matrix, row, out);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <::vext::Op Kp>
void
BM_EigenCpuBroadcast(
	benchmark::State& state)
{
	const std::vector<std::uint32_t> shape         = shape_from_state(state);
	const Eigen::Index               size          = shape_length(shape);
	const Eigen::Index               cols          = shape.back();
	const std::vector<float>         matrix_values = make_values(size, Kp == ::vext::Op::POW);
	const std::vector<float>         row_values    = make_values(cols, true);
	std::vector<float>               broadcast_values(static_cast<std::size_t>(size));

	for(Eigen::Index index = 0; index < size; ++index)
		broadcast_values[static_cast<std::size_t>(index)] = row_values[static_cast<std::size_t>(index % cols)];

	const Eigen::Map<const Eigen::ArrayXf> matrix(matrix_values.data(), size);
	const Eigen::Map<const Eigen::ArrayXf> row(broadcast_values.data(), size);
	Eigen::ArrayXf                         out(size);

	for([[maybe_unused]] auto iteration : state)
		{
			run_eigen_binary<Kp>(out, matrix, row);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * size);
}

} // namespace vext::benchmarks::cpu::ops

#endif
