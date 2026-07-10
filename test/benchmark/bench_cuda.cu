#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include <vext/core/cuda/allocator.cuh>
#include <vext/core/cuda/operations/elementwise_binary.cuh>
#include <vext/core/cuda/operations/linear_algebra.cuh>
#include <vext/core/cuda/operations/memory.cuh>
#include <vext/core/cuda/operations/reduction.cuh>
#include <vext/tensor.hpp>

namespace
{

#if defined(__GNUC__) || defined(__clang__)
#define VEXT_BENCHMARK_NOINLINE __attribute__((noinline))
#else
#define VEXT_BENCHMARK_NOINLINE
#endif

bool
has_cuda_device()
{
	std::int32_t count = 0;
	cudaError_t  err   = cudaGetDeviceCount(&count);

	return err == cudaSuccess && count > 0;
}

VEXT_BENCHMARK_NOINLINE void
observe_device_buffer(
	std::uint8_t* ptr)
{
	benchmark::DoNotOptimize(ptr);
	benchmark::ClobberMemory();
}

class CudaEventTimer
{
public:
	CudaEventTimer()
	{
		cudaEventCreate(&__start);
		cudaEventCreate(&__stop);
	}

	~CudaEventTimer()
	{
		cudaEventDestroy(__stop);
		cudaEventDestroy(__start);
	}

public:
	void
	start() const
	{
		cudaEventRecord(__start, 0);
	}

	double
	stop_seconds() const
	{
		cudaEventRecord(__stop, 0);
		cudaEventSynchronize(__stop);

		float milliseconds = 0.0f;
		cudaEventElapsedTime(&milliseconds, __start, __stop);

		return static_cast<double>(milliseconds) / 1000.0;
	}

private:
	cudaEvent_t __start = nullptr;
	cudaEvent_t __stop  = nullptr;
};

template <typename Tp>
Tp*
copy_to_device(
	const std::vector<Tp>& host)
{
	Tp* device = vext::core::cuda::allocator::allocate<Tp>(host.size());
	vext::core::cuda::operations::memcpy<Tp, vext::Backend::CUDA, vext::Backend::CPU>(device, host.data(), static_cast<std::uint32_t>(host.size()));
	return device;
}

bool
skip_without_cuda_device(
	benchmark::State& state)
{
	if(!has_cuda_device())
		{
			state.SkipWithError("No CUDA-capable device is available");
			return true;
		}

	return false;
}

void
BM_CudaAllocatorSmall(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint64_t bytes = static_cast<std::uint64_t>(state.range(0));

	for([[maybe_unused]] auto iteration : state)
		{
			std::uint8_t* ptr = vext::core::cuda::allocator::allocate<std::uint8_t>(bytes);
			observe_device_buffer(ptr);
			vext::core::cuda::allocator::deallocate(ptr);
		}

	cudaDeviceSynchronize();
	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * bytes));
	vext::core::cuda::allocator::free();
}

void
BM_CudaAllocatorLarge(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint64_t bytes = static_cast<std::uint64_t>(state.range(0));

	for([[maybe_unused]] auto iteration : state)
		{
			std::uint8_t* ptr = vext::core::cuda::allocator::allocate<std::uint8_t>(bytes);
			observe_device_buffer(ptr);
			vext::core::cuda::allocator::deallocate(ptr);
		}

	cudaDeviceSynchronize();
	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * bytes));
	vext::core::cuda::allocator::free();
}

void
BM_CudaTensorConstruct(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint32_t rows = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t cols = static_cast<std::uint32_t>(state.range(1));

	for([[maybe_unused]] auto iteration : state)
		{
			vext::Tensor<float, vext::Backend::CUDA> tensor(rows, cols);
			benchmark::DoNotOptimize(tensor.shape().length());
		}

	cudaDeviceSynchronize();
	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * rows * cols));
}

void
BM_CudaElementwiseAddKernel(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint32_t  size = static_cast<std::uint32_t>(state.range(0));
	std::vector<float>   host_lhs(size, 1.25f);
	std::vector<float>   host_rhs(size, 2.0f);
	float*               lhs = copy_to_device(host_lhs);
	float*               rhs = copy_to_device(host_rhs);
	float*               out = vext::core::cuda::allocator::allocate<float>(size);
	const CudaEventTimer timer;

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			vext::core::cuda::operations::binary<vext::core::BinaryOperation::ADD>(out, lhs, rhs, size);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out);
		}

	cudaDeviceSynchronize();
	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * sizeof(float) * 3));

	vext::core::cuda::allocator::deallocate(lhs);
	vext::core::cuda::allocator::deallocate(rhs);
	vext::core::cuda::allocator::deallocate(out);
	vext::core::cuda::allocator::free();
}

void
BM_CudaTensorAdd(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint32_t                            rows = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t                            cols = static_cast<std::uint32_t>(state.range(1));
	const vext::Tensor<float, vext::Backend::CUDA> lhs(rows, cols);
	const vext::Tensor<float, vext::Backend::CUDA> rhs(rows, cols);
	const CudaEventTimer                           timer;

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			vext::Tensor<float, vext::Backend::CUDA> out = lhs + rhs;
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.shape().length());
		}

	cudaDeviceSynchronize();
	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * rows * cols));
	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * rows * cols * sizeof(float) * 3));
}

void
BM_CudaReductionSumKernel(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint32_t size = static_cast<std::uint32_t>(state.range(0));
	std::vector<float>  host_values(size, 1.25f);
	float*              values = copy_to_device(host_values);
	float*              out    = vext::core::cuda::allocator::allocate<float>(1);

	const std::vector<std::uint32_t> keep_dims{ 1 };
	const std::vector<std::uint32_t> keep_strides{ 0 };
	const std::vector<std::uint32_t> reduce_dims{ size };
	const std::vector<std::uint32_t> reduce_strides{ 1 };
	const CudaEventTimer             timer;

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			vext::core::cuda::operations::reduce<vext::core::ReductionOperation::SUM>(out, values, 1, size, keep_dims, keep_strides, reduce_dims, reduce_strides);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out);
		}

	cudaDeviceSynchronize();
	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * sizeof(float)));

	vext::core::cuda::allocator::deallocate(values);
	vext::core::cuda::allocator::deallocate(out);
	vext::core::cuda::allocator::free();
}

void
BM_CudaMatmulKernel(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint32_t m = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t p = static_cast<std::uint32_t>(state.range(1));
	const std::uint32_t n = static_cast<std::uint32_t>(state.range(2));

	float*               lhs = copy_to_device(std::vector<float>(m * p, 0.5f));
	float*               rhs = copy_to_device(std::vector<float>(p * n, 0.25f));
	float*               out = vext::core::cuda::allocator::allocate<float>(m * n);
	const CudaEventTimer timer;

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			vext::core::cuda::operations::matmul(out, lhs, rhs, m, p, n);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out);
		}

	cudaDeviceSynchronize();
	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * m * n));
	state.counters["flop"] = benchmark::Counter(static_cast<double>(state.iterations()) * 2.0 * m * n * p, benchmark::Counter::kIsRate);

	vext::core::cuda::allocator::deallocate(lhs);
	vext::core::cuda::allocator::deallocate(rhs);
	vext::core::cuda::allocator::deallocate(out);
	vext::core::cuda::allocator::free();
}

void
BM_CudaTensorMatmul(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint32_t m = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t p = static_cast<std::uint32_t>(state.range(1));
	const std::uint32_t n = static_cast<std::uint32_t>(state.range(2));

	const vext::Tensor<float, vext::Backend::CUDA> lhs(m, p);
	const vext::Tensor<float, vext::Backend::CUDA> rhs(p, n);
	const CudaEventTimer                           timer;

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			vext::Tensor<float, vext::Backend::CUDA> out = lhs.matmul(rhs);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.shape().length());
		}

	cudaDeviceSynchronize();
	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * m * n));
	state.counters["flop"] = benchmark::Counter(static_cast<double>(state.iterations()) * 2.0 * m * n * p, benchmark::Counter::kIsRate);
}

}

BENCHMARK(BM_CudaAllocatorSmall)->Arg(256)->Arg(4096)->Iterations(100000);
BENCHMARK(BM_CudaAllocatorLarge)->Arg(2 * 1024 * 1024)->Arg(32 * 1024 * 1024)->Iterations(16);
BENCHMARK(BM_CudaTensorConstruct)->Args({ 256, 256 })->Args({ 1024, 1024 })->Iterations(16);
BENCHMARK(BM_CudaElementwiseAddKernel)->Arg(1 << 16)->Arg(1 << 20)->Iterations(64)->UseManualTime();
BENCHMARK(BM_CudaTensorAdd)->Args({ 256, 256 })->Args({ 1024, 1024 })->Iterations(16)->UseManualTime();
BENCHMARK(BM_CudaReductionSumKernel)->Arg(1 << 16)->Arg(1 << 20)->Iterations(64)->UseManualTime();
BENCHMARK(BM_CudaMatmulKernel)->Args({ 64, 64, 64 })->Args({ 128, 128, 128 })->Iterations(8)->UseManualTime();
BENCHMARK(BM_CudaTensorMatmul)->Args({ 64, 64, 64 })->Args({ 128, 128, 128 })->Iterations(8)->UseManualTime();

BENCHMARK_MAIN();

#undef VEXT_BENCHMARK_NOINLINE
