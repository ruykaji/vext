#ifndef __VEXT_BENCHMARKS_CPU_COMMON_HPP__
#define __VEXT_BENCHMARKS_CPU_COMMON_HPP__

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include <vext/type.hpp>

namespace vext::benchmarks::cpu
{

// Retained for registration macro compatibility; shapes are defined below.
inline constexpr std::int64_t VECTOR_SIZE   = 0;
inline constexpr std::int64_t CSR_ROWS      = 4096;
inline constexpr std::int64_t CSR_DEGREE    = 16;
inline constexpr std::int64_t CSR_FEATURES  = 64;
inline constexpr float        UNARY_PARAM_A = 1.25f;
inline constexpr float        UNARY_PARAM_B = 2.0f;

inline void
apply_vector_sizes(
	benchmark::Benchmark* benchmark)
{
	benchmark->Args({ 1, 65536, 1, 1, 1 });
	benchmark->Args({ 1, 1048576, 1, 1, 1 });
	benchmark->Args({ 1, 16777216, 1, 1, 1 });
	benchmark->Args({ 2, 256, 256, 1, 1 });
	benchmark->Args({ 2, 1024, 1024, 1, 1 });
	benchmark->Args({ 2, 4096, 4096, 1, 1 });
	benchmark->Args({ 3, 4, 128, 128, 1 });
	benchmark->Args({ 3, 64, 128, 128, 1 });
	benchmark->Args({ 3, 64, 512, 512, 1 });
	benchmark->Args({ 4, 1, 16, 64, 64 });
	benchmark->Args({ 4, 16, 64, 32, 32 });
	benchmark->Args({ 4, 16, 256, 64, 64 });
}

inline void
apply_matrix_sizes(
	benchmark::Benchmark* benchmark)
{
	apply_vector_sizes(benchmark);
}

inline void
apply_axis_reduction_sizes(
	benchmark::Benchmark* benchmark)
{
	apply_vector_sizes(benchmark);
}

inline void
apply_matmul_sizes(
	benchmark::Benchmark* benchmark)
{
	benchmark->Args({ 128, 128, 128 });
	benchmark->Args({ 512, 512, 512 });
	benchmark->Args({ 2048, 2048, 2048 });
	benchmark->Args({ 64, 128, 32 });
	benchmark->Args({ 256, 512, 128 });
	benchmark->Args({ 1024, 2048, 512 });
}

inline void
apply_csr_scatter_sizes(benchmark::Benchmark* benchmark)
{
	benchmark->Args({ 1024, 8, 16 });
	benchmark->Args({ 4096, 16, 64 });
	benchmark->Args({ 16384, 32, 128 });
}

inline void
apply_csr_spmv_sizes(benchmark::Benchmark* benchmark)
{
	benchmark->Args({ 1024, 8 });
	benchmark->Args({ 4096, 16 });
	benchmark->Args({ 16384, 32 });
}

inline std::vector<std::uint32_t>
shape_from_state(
	benchmark::State& state)
{
	const std::int64_t         rank = state.range(0);
	std::vector<std::uint32_t> shape;
	shape.reserve(static_cast<std::size_t>(rank));

	for(std::int64_t dimension = 0; dimension < rank; ++dimension)
		shape.push_back(static_cast<std::uint32_t>(state.range(dimension + 1)));

	return shape;
}

inline std::uint32_t
shape_length(
	const std::vector<std::uint32_t>& shape)
{
	return std::accumulate(shape.begin(), shape.end(), std::uint32_t{ 1 }, std::multiplies<std::uint32_t>{});
}

inline std::vector<float>
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
inline std::vector<float>
make_unary_values(
	const std::int64_t size)
{
	return make_values(size, Kp == vext::Op::LOG || Kp == vext::Op::SQRT || Kp == vext::Op::POW);
}

template <vext::Op Kp>
inline std::vector<float>
make_reduction_values(
	const std::int64_t size)
{
	if constexpr(Kp == vext::Op::PROD)
		{
			std::vector<float> values(static_cast<std::size_t>(size));
			for(std::int64_t i = 0; i < size; ++i)
				values[static_cast<std::size_t>(i)] = 1.0f + static_cast<float>(i % 17 - 8) * 0.00001f;
			return values;
		}

	return make_values(size);
}

}

#endif
