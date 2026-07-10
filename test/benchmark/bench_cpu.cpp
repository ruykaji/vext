#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include <vext/core/cpu/allocator.hpp>
#include <vext/core/cpu/operations/elementwise_binary.hpp>
#include <vext/core/cpu/operations/linear_algebra.hpp>
#include <vext/core/cpu/operations/reduction.hpp>
#include <vext/tensor.hpp>

namespace
{

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
observe_cpu_tensor(
	const vext::Tensor<Tp>& tensor)
{
	benchmark::DoNotOptimize(tensor.item(0));
}

void
BM_CpuAllocatorSmall(
	benchmark::State& state)
{
	const std::uint64_t bytes = static_cast<std::uint64_t>(state.range(0));

	for([[maybe_unused]] auto iteration : state)
		{
			std::uint8_t* ptr = vext::core::cpu::allocator::allocate<std::uint8_t>(bytes);
			benchmark::DoNotOptimize(ptr);
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
			benchmark::DoNotOptimize(ptr);
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
	const std::uint32_t rows = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t cols = static_cast<std::uint32_t>(state.range(1));

	for([[maybe_unused]] auto iteration : state)
		{
			vext::Tensor<float> tensor(rows, cols);
			observe_cpu_tensor(tensor);
		}

	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * rows * cols));
}

void
BM_CpuTensorCopy(
	benchmark::State& state)
{
	const std::uint32_t rows = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t cols = static_cast<std::uint32_t>(state.range(1));
	const vext::Tensor<float> source(rows, cols);

	for([[maybe_unused]] auto iteration : state)
		{
			vext::Tensor<float> copy(source);
			observe_cpu_tensor(copy);
		}

	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * rows * cols * sizeof(float)));
}

void
BM_CpuElementwiseAddKernel(
	benchmark::State& state)
{
	const std::uint32_t size = static_cast<std::uint32_t>(state.range(0));
	std::vector<float>  lhs(size, 1.25f);
	std::vector<float>  rhs(size, 2.0f);
	std::vector<float>  out(size, 0.0f);

	for([[maybe_unused]] auto iteration : state)
		{
			vext::core::cpu::operations::binary<vext::core::BinaryOperation::ADD>(out.data(), lhs.data(), rhs.data(), size);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * sizeof(float) * 3));
}

void
BM_CpuTensorAdd(
	benchmark::State& state)
{
	const std::uint32_t rows = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t cols = static_cast<std::uint32_t>(state.range(1));
	const vext::Tensor<float> lhs(rows, cols);
	const vext::Tensor<float> rhs(rows, cols);

	for([[maybe_unused]] auto iteration : state)
		{
			vext::Tensor<float> out = lhs + rhs;
			observe_cpu_tensor(out);
		}

	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * rows * cols));
	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * rows * cols * sizeof(float) * 3));
}

void
BM_CpuReductionSumKernel(
	benchmark::State& state)
{
	const std::uint32_t size = static_cast<std::uint32_t>(state.range(0));
	std::vector<float>  values(size, 1.25f);
	float               result = 0.0f;

	const std::vector<std::uint32_t> keep_dims{ 1 };
	const std::vector<std::uint32_t> keep_strides{ 0 };
	const std::vector<std::uint32_t> reduce_dims{ size };
	const std::vector<std::uint32_t> reduce_strides{ 1 };

	for([[maybe_unused]] auto iteration : state)
		{
			vext::core::cpu::operations::reduce<vext::core::ReductionOperation::SUM>(&result, values.data(), 1, size, keep_dims, keep_strides, reduce_dims, reduce_strides);
			benchmark::DoNotOptimize(result);
		}

	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * sizeof(float)));
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
	const std::uint32_t m = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t p = static_cast<std::uint32_t>(state.range(1));
	const std::uint32_t n = static_cast<std::uint32_t>(state.range(2));

	const vext::Tensor<float> lhs(m, p);
	const vext::Tensor<float> rhs(p, n);

	for([[maybe_unused]] auto iteration : state)
		{
			vext::Tensor<float> out = lhs.matmul(rhs);
			observe_cpu_tensor(out);
		}

	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * m * n));
	state.counters["flop"] = benchmark::Counter(static_cast<double>(state.iterations()) * 2.0 * m * n * p, benchmark::Counter::kIsRate);
}

}

BENCHMARK(BM_CpuAllocatorSmall)->Arg(256)->Arg(4096)->Iterations(100000);
BENCHMARK(BM_CpuAllocatorLarge)->Arg(2 * 1024 * 1024)->Arg(32 * 1024 * 1024)->Iterations(16);
BENCHMARK(BM_CpuTensorConstruct)->Args({ 256, 256 })->Args({ 1024, 1024 })->Iterations(16);
BENCHMARK(BM_CpuTensorCopy)->Args({ 256, 256 })->Args({ 1024, 1024 })->Iterations(16);
BENCHMARK(BM_CpuElementwiseAddKernel)->Arg(1 << 16)->Arg(1 << 20)->Iterations(64);
BENCHMARK(BM_CpuTensorAdd)->Args({ 256, 256 })->Args({ 1024, 1024 })->Iterations(16);
BENCHMARK(BM_CpuReductionSumKernel)->Arg(1 << 16)->Arg(1 << 20)->Iterations(64);
BENCHMARK(BM_CpuMatmulKernel)->Args({ 64, 64, 64 })->Args({ 128, 128, 128 })->Iterations(8);
BENCHMARK(BM_CpuTensorMatmul)->Args({ 64, 64, 64 })->Args({ 128, 128, 128 })->Iterations(8);

BENCHMARK_MAIN();

#undef VEXT_BENCHMARK_NOINLINE
