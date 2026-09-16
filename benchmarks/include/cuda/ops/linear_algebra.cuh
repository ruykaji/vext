#ifndef __VEXT_BENCHMARKS_CUDA_LINEAR_ALGEBRA_CUH__
#define __VEXT_BENCHMARKS_CUDA_LINEAR_ALGEBRA_CUH__

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_transform.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <vext/ops.hpp>
#include <vext/tensor.hpp>

#include <cuda/buffer.cuh>
#include <cuda/common.cuh>
#include <cuda/timing.cuh>

namespace vext::benchmarks::cuda::ops
{

void
BM_VextCudaMatmul(
	benchmark::State& state)
{
	if(::vext::benchmarks::cuda::skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint32_t rows   = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t shared = static_cast<std::uint32_t>(state.range(1));
	const std::uint32_t cols   = static_cast<std::uint32_t>(state.range(2));

	::vext::Tensor<float, ::vext::Backend::CUDA> lhs(rows, shared);
	::vext::Tensor<float, ::vext::Backend::CUDA> rhs(shared, cols);
	::vext::Tensor<float, ::vext::Backend::CUDA> out(rows, cols);

	const ::vext::benchmarks::cuda::CudaEventTimer timer;

	lhs.set_from(make_values(static_cast<std::int64_t>(rows) * shared));
	rhs.set_from(make_values(static_cast<std::int64_t>(shared) * cols));
	::vext::matmul(lhs, rhs, out);

	{
		const cudaError_t status = cudaDeviceSynchronize();
		::vext::benchmarks::cuda::check_cuda(status, "Warming up vext CUDA matrix multiplication");
	}

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			::vext::matmul(lhs, rhs, out);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.counters["FLOP/s"] = benchmark::Counter(static_cast<double>(state.iterations()) * 2.0 * rows * shared * cols, benchmark::Counter::kIsRate);
}

void
BM_CublasCudaMatmul(
	benchmark::State& state)
{
	if(::vext::benchmarks::cuda::skip_without_cuda_device(state))
		{
			return;
		}

	const std::int32_t rows   = static_cast<std::int32_t>(state.range(0));
	const std::int32_t shared = static_cast<std::int32_t>(state.range(1));
	const std::int32_t cols   = static_cast<std::int32_t>(state.range(2));

	::vext::benchmarks::cuda::DeviceBuffer<float> lhs(static_cast<std::int64_t>(rows) * shared);
	::vext::benchmarks::cuda::DeviceBuffer<float> rhs(static_cast<std::int64_t>(shared) * cols);
	::vext::benchmarks::cuda::DeviceBuffer<float> out(static_cast<std::int64_t>(rows) * cols);

	const ::vext::benchmarks::cuda::CublasHandle   handle;
	const ::vext::benchmarks::cuda::CudaEventTimer timer;

	const float alpha = 1.0f;
	const float beta  = 0.0f;

	lhs.fill_from_host(make_values(static_cast<std::int64_t>(rows) * shared));
	rhs.fill_from_host(make_values(static_cast<std::int64_t>(shared) * cols));

	::vext::benchmarks::cuda::check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, shared, &alpha, rhs.data(), cols, lhs.data(), shared, &beta, out.data(), cols), "Warming up cuBLAS SGEMM");

	{
		const cudaError_t status = cudaDeviceSynchronize();
		::vext::benchmarks::cuda::check_cuda(status, "Synchronizing cuBLAS SGEMM warmup");
	}

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			::vext::benchmarks::cuda::check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, shared, &alpha, rhs.data(), cols, lhs.data(), shared, &beta, out.data(), cols), "Running cuBLAS SGEMM");
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.counters["FLOP/s"] = benchmark::Counter(static_cast<double>(state.iterations()) * 2.0 * rows * shared * cols, benchmark::Counter::kIsRate);
}

} // namespace vext::benchmarks::cuda::ops

#endif
