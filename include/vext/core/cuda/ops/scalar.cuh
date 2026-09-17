#ifndef __VEXT_CORE_CUDA_SCALAR_CUH__
#define __VEXT_CORE_CUDA_SCALAR_CUH__

#include <iostream>

#include <cuda/std/cmath>
#include <cuda_runtime.h>

#include <vext/core/cuda/noise.cuh>
#include <vext/core/type.hpp>
#include <vext/type.hpp>

#define CUDA_CHECK(call)                                                                                            \
	do                                                                                                               \
		{                                                                                                             \
			cudaError_t err = (call);                                                                                  \
			if(err != cudaSuccess)                                                                                     \
				{                                                                                                       \
					std::cerr << __FILE__ << ":" << __LINE__ << " CUDA Error: " << cudaGetErrorString(err) << std::endl; \
					std::exit(EXIT_FAILURE);                                                                             \
				}                                                                                                       \
		}                                                                                                             \
	while(0)

namespace vext::core::cuda::ops::kernel
{

template <Op Kp, EvaluationMode Mp, typename T1, typename T2, typename T3, typename Dp = core::no_value_t>
__global__ void
scalar(
	T1*                 out,
	T2*                 src,
	const std::uint32_t N,
	const T3            value,
	const Dp            maybe_descriptor)
{
	const std::uint32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
	const std::uint32_t stride = blockDim.x * gridDim.x;

	for(std::uint32_t i = tid; i < N; i += stride)
		{
			if constexpr(Kp == Op::ADD)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = (src[i] + noise(maybe_descriptor, i)) + value;
						}
					else
						{
							out[i] = src[i] + value;
						}
				}
			else if constexpr(Kp == Op::SUB)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = (src[i] + noise(maybe_descriptor, i)) - value;
						}
					else
						{
							out[i] = src[i] - value;
						}
				}
			else if constexpr(Kp == Op::MUL)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = (src[i] + noise(maybe_descriptor, i)) * value;
						}
					else
						{
							out[i] = src[i] * value;
						}
				}
			else if constexpr(Kp == Op::DIV)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = (src[i] + noise(maybe_descriptor, i)) / value;
						}
					else
						{
							out[i] = src[i] / value;
						}
				}
			else if constexpr(Kp == Op::POW)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = ::cuda::std::pow(src[i] + noise(maybe_descriptor, i), value);
						}
					else
						{
							out[i] = ::cuda::std::pow(src[i], value);
						}
				}
		}
}

}

namespace vext::core::cuda::ops
{

template <Op Kp, EvaluationMode Mp, typename T1, typename T2>
void
scalar(
	T1*                 src,
	const std::uint32_t N,
	const T2            value)
{
	constexpr std::uint32_t block_size = 256;
	const std::uint32_t     grid_size  = (N + block_size - 1) / block_size;

	if constexpr(Mp == EvaluationMode::PERTURBED)
		{
			const NoiseDescriptor& descriptor = sequentional_noise_descriptor();
			kernel::scalar<Kp, Mp><<<grid_size, block_size>>>(src, src, N, descriptor);
			CUDA_CHECK(cudaGetLastError());
		}
	else
		{
			kernel::scalar<Kp, Mp><<<grid_size, block_size>>>(src, src, N, core::no_value);
			CUDA_CHECK(cudaGetLastError());
		}
}

}

#endif
