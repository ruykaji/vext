#ifndef __VEXT_BENCHMARKS_CPU_LINEAR_ALGEBRA_HPP__
#define __VEXT_BENCHMARKS_CPU_LINEAR_ALGEBRA_HPP__

#include <benchmark/benchmark.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <vext/ops.hpp>
#include <vext/tensor.hpp>

#include <cpu/common.hpp>

namespace vext::benchmarks::cpu::ops
{

void
BM_VextCpuMatmul(
	benchmark::State& state)
{
	const std::uint32_t      rows   = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t      shared = static_cast<std::uint32_t>(state.range(1));
	const std::uint32_t      cols   = static_cast<std::uint32_t>(state.range(2));
	const std::vector<float> zeros(static_cast<std::size_t>(rows) * cols, 0.0f);

	::vext::Tensor<float> lhs(rows, shared);
	::vext::Tensor<float> rhs(shared, cols);
	::vext::Tensor<float> out(rows, cols);
	lhs.set_from(make_values(static_cast<std::int64_t>(rows) * shared));
	rhs.set_from(make_values(static_cast<std::int64_t>(shared) * cols));

	for([[maybe_unused]] auto iteration : state)
		{
			state.PauseTiming();
			out.set_from(zeros);
			state.ResumeTiming();
			::vext::matmul(lhs, rhs, out);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.counters["FLOP/s"] = benchmark::Counter(static_cast<double>(state.iterations()) * 2.0 * rows * shared * cols, benchmark::Counter::kIsRate);
}

void
BM_EigenCpuMatmul(
	benchmark::State& state)
{
	using Matrix                        = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	const Eigen::Index       rows       = state.range(0);
	const Eigen::Index       shared     = state.range(1);
	const Eigen::Index       cols       = state.range(2);
	const std::vector<float> lhs_values = make_values(rows * shared);
	const std::vector<float> rhs_values = make_values(shared * cols);

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

} // namespace vext::benchmarks::cpu::ops

#endif
