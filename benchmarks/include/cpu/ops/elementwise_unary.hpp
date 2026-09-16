#ifndef __VEXT_BENCHMARKS_CPU_ELEMENTWISE_UNARY_HPP__
#define __VEXT_BENCHMARKS_CPU_ELEMENTWISE_UNARY_HPP__

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
run_vext_unary(
	::vext::Tensor<float>& values)
{
	if constexpr(Kp == ::vext::Op::LEAKY_RELU || Kp == ::vext::Op::ELU || Kp == ::vext::Op::SWISH)
		{
			::vext::unary<Kp>(values, UNARY_PARAM_A);
		}
	else if constexpr(Kp == ::vext::Op::LINEAR || Kp == ::vext::Op::CLIP || Kp == ::vext::Op::POW)
		{
			::vext::unary<Kp>(values, UNARY_PARAM_A, UNARY_PARAM_B);
		}
	else
		{
			::vext::unary<Kp>(values);
		}
}

template <::vext::Op Kp>
void
run_eigen_unary(
	Eigen::ArrayXf& values)
{
	if constexpr(Kp == ::vext::Op::ABS)
		{
			values = values.abs();
		}
	else if constexpr(Kp == ::vext::Op::SIN)
		{
			values = values.sin();
		}
	else if constexpr(Kp == ::vext::Op::COS)
		{
			values = values.cos();
		}
	else if constexpr(Kp == ::vext::Op::TANH)
		{
			values = values.tanh();
		}
	else if constexpr(Kp == ::vext::Op::NEG)
		{
			values = -values;
		}
	else if constexpr(Kp == ::vext::Op::EXP)
		{
			values = values.exp();
		}
	else if constexpr(Kp == ::vext::Op::LOG)
		{
			values = values.log();
		}
	else if constexpr(Kp == ::vext::Op::SQRT)
		{
			values = values.sqrt();
		}
	else if constexpr(Kp == ::vext::Op::SQUARE)
		{
			values = values.square();
		}
	else if constexpr(Kp == ::vext::Op::ROUND)
		{
			values = values.round();
		}
	else if constexpr(Kp == ::vext::Op::SIGMOID)
		{
			values = 1.0f / (1.0f + (-values).exp());
		}
	else if constexpr(Kp == ::vext::Op::SOFT_RELU)
		{
			values = (1.0f + values.exp()).log();
		}
	else if constexpr(Kp == ::vext::Op::RELU)
		{
			values = values.max(0.0f);
		}
	else if constexpr(Kp == ::vext::Op::SOFTMAX)
		{
			values = values.exp();
			values /= values.sum();
		}
	else if constexpr(Kp == ::vext::Op::SOFTMIN)
		{
			values = (-values).exp();
			values /= values.sum();
		}
	else if constexpr(Kp == ::vext::Op::LOGSOFTMAX)
		{
			values = values.exp();
			values = (values / values.sum()).log();
		}
	else if constexpr(Kp == ::vext::Op::LEAKY_RELU)
		{
			values = values.max(0.0f) + UNARY_PARAM_A * values.min(0.0f);
		}
	else if constexpr(Kp == ::vext::Op::ELU)
		{
			values = (values > 0.0f).select(values, UNARY_PARAM_A * (values.exp() - 1.0f));
		}
	else if constexpr(Kp == ::vext::Op::SWISH)
		{
			values = values / (1.0f + (-UNARY_PARAM_A * values).exp());
		}
	else if constexpr(Kp == ::vext::Op::LINEAR)
		{
			values = UNARY_PARAM_A * values + UNARY_PARAM_B;
		}
	else if constexpr(Kp == ::vext::Op::CLIP)
		{
			values = values.max(UNARY_PARAM_A).min(UNARY_PARAM_B);
		}
	else if constexpr(Kp == ::vext::Op::POW)
		{
			values = UNARY_PARAM_A * values.pow(UNARY_PARAM_B);
		}
}

template <::vext::Op Kp>
void
BM_VextCpuUnary(
	benchmark::State& state)
{
	const std::vector<std::uint32_t> shape = shape_from_state(state);
	const std::uint32_t              size  = shape_length(shape);
	const std::vector<float>         input = make_unary_values<Kp>(size);

	::vext::Tensor<float> values(shape);
	values.set_from(input);

	for([[maybe_unused]] auto iteration : state)
		{
			state.PauseTiming();
			values.set_from(input);
			state.ResumeTiming();
			run_vext_unary<Kp>(values);
			benchmark::DoNotOptimize(values.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <::vext::Op Kp>
void
BM_EigenCpuUnary(
	benchmark::State& state)
{
	const std::vector<std::uint32_t>       shape        = shape_from_state(state);
	const Eigen::Index                     size         = shape_length(shape);
	const std::vector<float>               input_values = make_unary_values<Kp>(size);
	const Eigen::Map<const Eigen::ArrayXf> input(input_values.data(), size);

	Eigen::ArrayXf values = input;

	for([[maybe_unused]] auto iteration : state)
		{
			state.PauseTiming();
			values = input;
			state.ResumeTiming();
			run_eigen_unary<Kp>(values);
			benchmark::DoNotOptimize(values.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * size);
}

} // namespace vext::benchmarks::cpu::ops

#endif
