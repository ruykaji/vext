#ifndef __VEXT_BENCHMARKS_CUDA_ELEMENTWISE_BINARY_CUH__
#define __VEXT_BENCHMARKS_CUDA_ELEMENTWISE_BINARY_CUH__

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

template <::vext::Op Kp>
struct BinaryFunctor
{
	__host__ __device__ float
	operator()(
		const float lhs,
		const float rhs) const
	{
		if constexpr(Kp == ::vext::Op::ADD)
			{
				return lhs + rhs;
			}
		else if constexpr(Kp == ::vext::Op::SUB)
			{
				return lhs - rhs;
			}
		else if constexpr(Kp == ::vext::Op::MUL)
			{
				return lhs * rhs;
			}
		else if constexpr(Kp == ::vext::Op::DIV)
			{
				return lhs / rhs;
			}
		else if constexpr(Kp == ::vext::Op::POW)
			{
				return powf(lhs, rhs);
			}
		else if constexpr(Kp == ::vext::Op::MIN)
			{
				return fminf(lhs, rhs);
			}
		else if constexpr(Kp == ::vext::Op::MAX)
			{
				return fmaxf(lhs, rhs);
			}
		else
			{
				return fmaxf(0.0f, lhs) + rhs * fminf(0.0f, lhs);
			}
	}
};

template <::vext::Op Kp>
struct BroadcastFunctor
{
	const float* matrix;
	const float* row;

	std::int32_t cols;

	__host__ __device__ float
	operator()(
		const std::int32_t index) const
	{
		return BinaryFunctor<Kp>{}(matrix[index], row[index % cols]);
	}
};

template <::vext::Op Kp>
void
BM_VextCudaBinary(
	benchmark::State& state)
{
	if(::vext::benchmarks::cuda::skip_without_cuda_device(state))
		{
			return;
		}

	const std::vector<std::uint32_t> shape = shape_from_state(state);
	const std::uint32_t              size  = shape_length(shape);

	::vext::Tensor<float, ::vext::Backend::CUDA> lhs(shape);
	::vext::Tensor<float, ::vext::Backend::CUDA> rhs(shape);
	::vext::Tensor<float, ::vext::Backend::CUDA> out(shape);

	const ::vext::benchmarks::cuda::CudaEventTimer timer;

	lhs.set_from(make_values(size, Kp == ::vext::Op::POW));
	rhs.set_from(make_values(size, true));
	::vext::binary<Kp>(lhs, rhs, out);

	{
		const cudaError_t status = cudaDeviceSynchronize();
		::vext::benchmarks::cuda::check_cuda(status, "Warming up vext CUDA binary operation");
	}

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			::vext::binary<Kp>(lhs, rhs, out);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <::vext::Op Kp>
void
BM_CubCudaBinary(
	benchmark::State& state)
{
	if(::vext::benchmarks::cuda::skip_without_cuda_device(state))
		{
			return;
		}

	const std::int32_t size = static_cast<std::int32_t>(shape_length(shape_from_state(state)));

	::vext::benchmarks::cuda::DeviceBuffer<float> lhs(size);
	::vext::benchmarks::cuda::DeviceBuffer<float> rhs(size);
	::vext::benchmarks::cuda::DeviceBuffer<float> out(size);

	const ::vext::benchmarks::cuda::CudaEventTimer timer;

	lhs.fill_from_host(make_values(size, Kp == ::vext::Op::POW));
	rhs.fill_from_host(make_values(size, true));

	{
		const cudaError_t status = cub::DeviceTransform::Transform(
			::cuda::std::make_tuple(lhs.data(), rhs.data()),
			out.data(),
			size,
			BinaryFunctor<Kp>{});
		::vext::benchmarks::cuda::check_cuda(status, "Warming up CUB CUDA binary operation");
	}

	{
		const cudaError_t status = cudaDeviceSynchronize();
		::vext::benchmarks::cuda::check_cuda(status, "Synchronizing CUB CUDA binary warmup");
	}

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();

			{
				const cudaError_t status = cub::DeviceTransform::Transform(
					::cuda::std::make_tuple(lhs.data(), rhs.data()),
					out.data(),
					size,
					BinaryFunctor<Kp>{});
				::vext::benchmarks::cuda::check_cuda(status, "Running CUB CUDA binary operation");
			}

			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <::vext::Op Kp>
void
BM_VextCudaBroadcast(
	benchmark::State& state)
{
	if(::vext::benchmarks::cuda::skip_without_cuda_device(state))
		{
			return;
		}

	const std::vector<std::uint32_t> shape = shape_from_state(state);
	const std::uint32_t              size  = shape_length(shape);
	const std::uint32_t              cols  = shape.back();

	::vext::Tensor<float, ::vext::Backend::CUDA> matrix(shape);
	::vext::Tensor<float, ::vext::Backend::CUDA> row(cols);
	::vext::Tensor<float, ::vext::Backend::CUDA> out(shape);

	const ::vext::benchmarks::cuda::CudaEventTimer timer;

	matrix.set_from(make_values(size, Kp == ::vext::Op::POW));
	row.set_from(make_values(cols, true));
	::vext::binary<Kp>(matrix, row, out);

	{
		const cudaError_t status = cudaDeviceSynchronize();
		::vext::benchmarks::cuda::check_cuda(status, "Warming up vext CUDA broadcast operation");
	}

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			::vext::binary<Kp>(matrix, row, out);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <::vext::Op Kp>
void
BM_CubCudaBroadcast(
	benchmark::State& state)
{
	if(::vext::benchmarks::cuda::skip_without_cuda_device(state))
		{
			return;
		}

	const std::vector<std::uint32_t> shape = shape_from_state(state);
	const std::int32_t               size  = static_cast<std::int32_t>(shape_length(shape));
	const std::int32_t               cols  = static_cast<std::int32_t>(shape.back());

	::vext::benchmarks::cuda::DeviceBuffer<float> matrix(size);
	::vext::benchmarks::cuda::DeviceBuffer<float> row(cols);
	::vext::benchmarks::cuda::DeviceBuffer<float> out(size);

	const ::vext::benchmarks::cuda::CudaEventTimer timer;

	matrix.fill_from_host(make_values(size, Kp == ::vext::Op::POW));
	row.fill_from_host(make_values(cols, true));

	{
		const cudaError_t status = cub::DeviceTransform::Transform(
			thrust::make_counting_iterator<std::int32_t>(0),
			out.data(),
			size,
			BroadcastFunctor<Kp>{ matrix.data(), row.data(), cols });
		::vext::benchmarks::cuda::check_cuda(status, "Warming up CUB CUDA broadcast operation");
	}

	{
		const cudaError_t status = cudaDeviceSynchronize();
		::vext::benchmarks::cuda::check_cuda(status, "Synchronizing CUB CUDA broadcast warmup");
	}

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();

			{
				const cudaError_t status = cub::DeviceTransform::Transform(
					thrust::make_counting_iterator<std::int32_t>(0),
					out.data(),
					size,
					BroadcastFunctor<Kp>{ matrix.data(), row.data(), cols });
				::vext::benchmarks::cuda::check_cuda(status, "Running CUB CUDA broadcast operation");
			}

			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * size);
}

} // namespace vext::benchmarks::cuda::ops

#endif
