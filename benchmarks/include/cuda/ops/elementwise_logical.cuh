#ifndef __VEXT_BENCHMARKS_CUDA_ELEMENTWISE_LOGICAL_CUH__
#define __VEXT_BENCHMARKS_CUDA_ELEMENTWISE_LOGICAL_CUH__

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
struct LogicalFunctor
{
	__host__ __device__
		std::uint8_t
		operator()(
			const float lhs,
			const float rhs) const
	{
		if constexpr(Kp == ::vext::Op::EQUAL)
			{
				return lhs == rhs;
			}
		else if constexpr(Kp == ::vext::Op::NOT_EQUAL)
			{
				return lhs != rhs;
			}
		else if constexpr(Kp == ::vext::Op::LESS)
			{
				return lhs < rhs;
			}
		else if constexpr(Kp == ::vext::Op::LESS_EQUAL)
			{
				return lhs <= rhs;
			}
		else if constexpr(Kp == ::vext::Op::GREATER)
			{
				return lhs > rhs;
			}
		else
			{
				return lhs >= rhs;
			}
	}
};

template <::vext::Op Kp>
void
BM_VextCudaLogical(
	benchmark::State& state)
{
	if(::vext::benchmarks::cuda::skip_without_cuda_device(state))
		{
			return;
		}

	const std::vector<std::uint32_t> shape = shape_from_state(state);
	const std::uint32_t              size  = shape_length(shape);

	::vext::Tensor<float, ::vext::Backend::CUDA>        lhs(shape);
	::vext::Tensor<float, ::vext::Backend::CUDA>        rhs(shape);
	::vext::Tensor<std::uint8_t, ::vext::Backend::CUDA> out(shape);

	const ::vext::benchmarks::cuda::CudaEventTimer timer;

	lhs.set_from(make_values(size));
	rhs.set_from(make_values(size, true));
	::vext::logical<Kp>(lhs, rhs, out);

	{
		const cudaError_t status = cudaDeviceSynchronize();
		::vext::benchmarks::cuda::check_cuda(status, "Warming up vext CUDA logical operation");
	}

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			::vext::logical<Kp>(lhs, rhs, out);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <::vext::Op Kp>
void
BM_CubCudaLogical(
	benchmark::State& state)
{
	if(::vext::benchmarks::cuda::skip_without_cuda_device(state))
		{
			return;
		}

	const std::int32_t size = static_cast<std::int32_t>(shape_length(shape_from_state(state)));

	::vext::benchmarks::cuda::DeviceBuffer<float>        lhs(size);
	::vext::benchmarks::cuda::DeviceBuffer<float>        rhs(size);
	::vext::benchmarks::cuda::DeviceBuffer<std::uint8_t> out(size);

	const ::vext::benchmarks::cuda::CudaEventTimer timer;

	lhs.fill_from_host(make_values(size));
	rhs.fill_from_host(make_values(size, true));

	{
		const cudaError_t status = cub::DeviceTransform::Transform(
			::cuda::std::make_tuple(lhs.data(), rhs.data()),
			out.data(),
			size,
			LogicalFunctor<Kp>{});
		::vext::benchmarks::cuda::check_cuda(status, "Warming up CUB CUDA logical operation");
	}

	{
		const cudaError_t status = cudaDeviceSynchronize();
		::vext::benchmarks::cuda::check_cuda(status, "Synchronizing CUB CUDA logical warmup");
	}

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();

			{
				const cudaError_t status = cub::DeviceTransform::Transform(
					::cuda::std::make_tuple(lhs.data(), rhs.data()),
					out.data(),
					size,
					LogicalFunctor<Kp>{});
				::vext::benchmarks::cuda::check_cuda(status, "Running CUB CUDA logical operation");
			}

			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * size);
}

} // namespace vext::benchmarks::cuda::ops

#endif
