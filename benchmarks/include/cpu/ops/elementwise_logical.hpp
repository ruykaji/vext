#ifndef __VEXT_BENCHMARKS_CPU_ELEMENTWISE_LOGICAL_HPP__
#define __VEXT_BENCHMARKS_CPU_ELEMENTWISE_LOGICAL_HPP__

#include <benchmark/benchmark.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <vext/ops.hpp>
#include <vext/tensor.hpp>

#include <cpu/common.hpp>

namespace vext::benchmarks::cpu::ops
{

template <::vext::Op Kp>
void
BM_VextCpuLogical(
	benchmark::State& state)
{
	const std::vector<std::uint32_t> shape = shape_from_state(state);
	const std::uint32_t              size  = shape_length(shape);

	::vext::Tensor<float>        lhs(shape);
	::vext::Tensor<float>        rhs(shape);
	::vext::Tensor<std::uint8_t> out(shape);

	lhs.set_from(make_values(size));
	rhs.set_from(make_values(size, true));

	for([[maybe_unused]] auto iteration : state)
		{
			::vext::logical<Kp>(lhs, rhs, out);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <::vext::Op Kp>
void
BM_EigenCpuLogical(
	benchmark::State& state)
{
	const std::vector<std::uint32_t> shape      = shape_from_state(state);
	const Eigen::Index               size       = shape_length(shape);
	const std::vector<float>         lhs_values = make_values(size);
	const std::vector<float>         rhs_values = make_values(size, true);

	const Eigen::Map<const Eigen::ArrayXf> lhs(lhs_values.data(), size);
	const Eigen::Map<const Eigen::ArrayXf> rhs(rhs_values.data(), size);

	Eigen::Array<bool, Eigen::Dynamic, 1> out(size);

	for([[maybe_unused]] auto iteration : state)
		{
			if constexpr(Kp == ::vext::Op::EQUAL)
				{
					out = lhs == rhs;
				}
			else if constexpr(Kp == ::vext::Op::NOT_EQUAL)
				{
					out = lhs != rhs;
				}
			else if constexpr(Kp == ::vext::Op::LESS)
				{
					out = lhs < rhs;
				}
			else if constexpr(Kp == ::vext::Op::LESS_EQUAL)
				{
					out = lhs <= rhs;
				}
			else if constexpr(Kp == ::vext::Op::GREATER)
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

} // namespace vext::benchmarks::cpu::ops

#endif
