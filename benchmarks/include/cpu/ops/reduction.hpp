#ifndef __VEXT_BENCHMARKS_CPU_REDUCTION_HPP__
#define __VEXT_BENCHMARKS_CPU_REDUCTION_HPP__

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
float
eigen_reduce(
	const Eigen::ArrayXf& values)
{
	if constexpr(Kp == ::vext::Op::SUM)
		{
			return values.sum();
		}
	else if constexpr(Kp == ::vext::Op::MEAN)
		{
			return values.mean();
		}
	else if constexpr(Kp == ::vext::Op::MIN)
		{
			return values.minCoeff();
		}
	else if constexpr(Kp == ::vext::Op::MAX)
		{
			return values.maxCoeff();
		}
	else if constexpr(Kp == ::vext::Op::PROD)
		{
			return values.prod();
		}
	else if constexpr(Kp == ::vext::Op::L2_NORM)
		{
			return std::sqrt(values.square().sum());
		}
	else
		{
			const float variance = (values - values.mean()).square().mean();
			return Kp == ::vext::Op::STD ? std::sqrt(variance) : variance;
		}
}

template <::vext::Op Kp>
void
BM_VextCpuReduction(
	benchmark::State& state)
{
	const std::vector<std::uint32_t> shape = shape_from_state(state);
	const std::uint32_t              size  = shape_length(shape);

	::vext::Tensor<float> values(shape);
	::vext::Tensor<float> out(1);

	values.set_from(make_reduction_values<Kp>(size));

	for([[maybe_unused]] auto iteration : state)
		{
			::vext::reduction<Kp>(values, ::vext::core::no_value_t{}, out);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <::vext::Op Kp>
void
BM_EigenCpuReduction(
	benchmark::State& state)
{
	const std::vector<std::uint32_t> shape = shape_from_state(state);
	const Eigen::Index               size  = shape_length(shape);
	const std::vector<float>         input = make_reduction_values<Kp>(size);

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

template <::vext::Op Kp>
void
BM_VextCpuAxisReduction(
	benchmark::State& state)
{
	const std::vector<std::uint32_t> shape = shape_from_state(state);
	const std::uint32_t              size  = shape_length(shape);
	const std::uint32_t              cols  = shape.back();
	std::vector<std::uint32_t>       output_shape(shape.begin(), shape.end() - 1);

	if(output_shape.empty())
		output_shape.push_back(1);

	::vext::Tensor<float> values(shape);
	::vext::Tensor<float> out(output_shape);

	values.set_from(make_reduction_values<Kp>(size));

	for([[maybe_unused]] auto iteration : state)
		{
			::vext::reduction<Kp>(values, ::vext::axes({ static_cast<std::int32_t>(shape.size() - 1) }), out);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <::vext::Op Kp>
void
BM_EigenCpuAxisReduction(
	benchmark::State& state)
{
	const std::vector<std::uint32_t>       shape = shape_from_state(state);
	const Eigen::Index                     size  = shape_length(shape);
	const Eigen::Index                     cols  = shape.back();
	const Eigen::Index                     rows  = size / cols;
	const std::vector<float>               input = make_reduction_values<Kp>(size);
	const Eigen::Map<const Eigen::ArrayXf> values(input.data(), size);
	Eigen::ArrayXf                         out(rows);

	for([[maybe_unused]] auto iteration : state)
		{
			for(Eigen::Index row = 0; row < rows; ++row)
				out[row] = eigen_reduce<Kp>(values.segment(row * cols, cols));

			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * size);
}

} // namespace vext::benchmarks::cpu::ops

#endif
