#ifndef __VEXT_BENCHMARKS_CUDA_ELEMENTWISE_UNARY_CUH__
#define __VEXT_BENCHMARKS_CUDA_ELEMENTWISE_UNARY_CUH__

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
struct UnaryFunctor
{
	__host__ __device__ float
	operator()(
		const float value) const
	{
		if constexpr(Kp == ::vext::Op::ABS)
			{
				return fabsf(value);
			}
		else if constexpr(Kp == ::vext::Op::SIN)
			{
				return sinf(value);
			}
		else if constexpr(Kp == ::vext::Op::COS)
			{
				return cosf(value);
			}
		else if constexpr(Kp == ::vext::Op::TANH)
			{
				return tanhf(value);
			}
		else if constexpr(Kp == ::vext::Op::NEG)
			{
				return -value;
			}
		else if constexpr(Kp == ::vext::Op::EXP)
			{
				return expf(value);
			}
		else if constexpr(Kp == ::vext::Op::LOG)
			{
				return logf(value);
			}
		else if constexpr(Kp == ::vext::Op::SQRT)
			{
				return sqrtf(value);
			}
		else if constexpr(Kp == ::vext::Op::SQUARE)
			{
				return value * value;
			}
		else if constexpr(Kp == ::vext::Op::ROUND)
			{
				return roundf(value);
			}
		else if constexpr(Kp == ::vext::Op::SIGMOID)
			{
				return 1.0f / (1.0f + expf(-value));
			}
		else if constexpr(Kp == ::vext::Op::SOFT_RELU)
			{
				return logf(1.0f + expf(value));
			}
		else if constexpr(Kp == ::vext::Op::RELU)
			{
				return value > 0.0f ? value : 0.0f;
			}
		else if constexpr(Kp == ::vext::Op::LEAKY_RELU)
			{
				return value > 0.0f ? value : UNARY_PARAM_A * value;
			}
		else if constexpr(Kp == ::vext::Op::ELU)
			{
				return value > 0.0f ? value : UNARY_PARAM_A * (expf(value) - 1.0f);
			}
		else if constexpr(Kp == ::vext::Op::SWISH)
			{
				return value / (1.0f + expf(-UNARY_PARAM_A * value));
			}
		else if constexpr(Kp == ::vext::Op::LINEAR)
			{
				return UNARY_PARAM_A * value + UNARY_PARAM_B;
			}
		else if constexpr(Kp == ::vext::Op::CLIP)
			{
				return fmaxf(UNARY_PARAM_A, fminf(UNARY_PARAM_B, value));
			}
		else if constexpr(Kp == ::vext::Op::POW)
			{
				return UNARY_PARAM_A * powf(value, UNARY_PARAM_B);
			}
		else if constexpr(Kp == ::vext::Op::SOFTMIN)
			{
				return expf(-value);
			}
		else
			{
				return expf(value);
			}
	}
};

template <::vext::Op Kp>
struct NormalizeFunctor
{
	const float* sum;

	__host__ __device__ float
	operator()(
		const float value) const
	{
		if constexpr(Kp == ::vext::Op::LOGSOFTMAX)
			{
				return logf(value / *sum);
			}
		else
			{
				return value / *sum;
			}
	}
};

template <::vext::Op Kp>
void
run_vext_unary(
	::vext::Tensor<float, ::vext::Backend::CUDA>& values)
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
run_cub_unary(
	::vext::benchmarks::cuda::DeviceBuffer<float>&        values,
	::vext::benchmarks::cuda::DeviceBuffer<float>&        sum,
	::vext::benchmarks::cuda::DeviceBuffer<std::uint8_t>& workspace,
	std::uint64_t&                                        workspace_size,
	const std::int32_t                                    size)
{
	{
		const cudaError_t status = cub::DeviceTransform::Transform(
			values.data(),
			values.data(),
			size,
			UnaryFunctor<Kp>{});
		::vext::benchmarks::cuda::check_cuda(status, "Running CUB unary transform");
	}

	if constexpr(Kp == ::vext::Op::SOFTMAX || Kp == ::vext::Op::SOFTMIN || Kp == ::vext::Op::LOGSOFTMAX)
		{
			{
				const cudaError_t status = cub::DeviceReduce::Sum(
					workspace.data(),
					workspace_size,
					values.data(),
					sum.data(),
					size);
				::vext::benchmarks::cuda::check_cuda(status, "Running CUB unary normalization reduction");
			}
			{
				const cudaError_t status = cub::DeviceTransform::Transform(
					values.data(),
					values.data(),
					size,
					NormalizeFunctor<Kp>{ sum.data() });
				::vext::benchmarks::cuda::check_cuda(status, "Running CUB unary normalization transform");
			}
		}
}

template <::vext::Op Kp>
void
BM_VextCudaUnary(
	benchmark::State& state)
{
	if(::vext::benchmarks::cuda::skip_without_cuda_device(state))
		{
			return;
		}

	const std::vector<std::uint32_t> shape = shape_from_state(state);
	const std::uint32_t              size  = shape_length(shape);
	const std::vector<float>         input = make_unary_values<Kp>(size);

	::vext::Tensor<float, ::vext::Backend::CUDA> values(shape);

	const ::vext::benchmarks::cuda::CudaEventTimer timer;

	values.set_from(input);
	run_vext_unary<Kp>(values);

	{
		const cudaError_t status = cudaDeviceSynchronize();
		::vext::benchmarks::cuda::check_cuda(status, "Warming up vext CUDA unary operation");
	}

	for([[maybe_unused]] auto iteration : state)
		{
			values.set_from(input);
			timer.start();
			run_vext_unary<Kp>(values);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(values.data());
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <::vext::Op Kp>
void
BM_CubCudaUnary(
	benchmark::State& state)
{
	if(::vext::benchmarks::cuda::skip_without_cuda_device(state))
		{
			return;
		}

	const std::int32_t size = static_cast<std::int32_t>(shape_length(shape_from_state(state)));

	::vext::benchmarks::cuda::DeviceBuffer<float> values(size);
	::vext::benchmarks::cuda::DeviceBuffer<float> sum(1);

	std::uint64_t workspace_size = 1;

	if constexpr(Kp == ::vext::Op::SOFTMAX || Kp == ::vext::Op::SOFTMIN || Kp == ::vext::Op::LOGSOFTMAX)
		{
			{
				const cudaError_t status = cub::DeviceReduce::Sum(
					nullptr,
					workspace_size,
					values.data(),
					sum.data(),
					size);
				::vext::benchmarks::cuda::check_cuda(status, "Querying CUB unary reduction workspace");
			}
		}

	::vext::benchmarks::cuda::DeviceBuffer<std::uint8_t> workspace(std::max<std::uint64_t>(workspace_size, 1));

	const ::vext::benchmarks::cuda::CudaEventTimer timer;
	const std::vector<float>                       input = make_unary_values<Kp>(size);

	values.fill_from_host(input);
	run_cub_unary<Kp>(values, sum, workspace, workspace_size, size);

	{
		const cudaError_t status = cudaDeviceSynchronize();
		::vext::benchmarks::cuda::check_cuda(status, "Warming up CUB CUDA unary operation");
	}

	for([[maybe_unused]] auto iteration : state)
		{
			values.fill_from_host(input);
			timer.start();
			run_cub_unary<Kp>(values, sum, workspace, workspace_size, size);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(values.data());
		}

	state.SetItemsProcessed(state.iterations() * size);
}

} // namespace vext::benchmarks::cuda::ops

#endif
