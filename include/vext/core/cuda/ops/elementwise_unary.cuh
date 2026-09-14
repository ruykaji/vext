#ifndef __VEXT_CORE_CUDA_OPS_ELEMENTWISE_UNARY_CUH__
#define __VEXT_CORE_CUDA_OPS_ELEMENTWISE_UNARY_CUH__

#include <iostream>

#include <cuda/std/cmath>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
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

template <typename Tp>
__global__ void
reduce_sum(
	Tp* __restrict__ partial,
	const Tp* __restrict__ x,
	const std::uint32_t N)
{
	Tp sum = 0;

	const std::uint32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
	const std::uint32_t stride = blockDim.x * gridDim.x;

	for(std::uint32_t i = tid; i < N; i += stride)
		{
			sum += x[i];
		}

	for(std::int32_t offset = warpSize / 2; offset > 0; offset >>= 1)
		{
			sum += __shfl_down_sync(0xffffffff, sum, offset);
		}

	__shared__ Tp warp_sums[32];

	const std::int32_t lane    = threadIdx.x % warpSize;
	const std::int32_t warp_id = threadIdx.x / warpSize;

	if(lane == 0)
		{
			warp_sums[warp_id] = sum;
		}

	__syncthreads();

	Tp block_sum = 0;

	if(warp_id == 0)
		{
			const std::int32_t warp_count = (blockDim.x + warpSize - 1) / warpSize;

			if(lane < warp_count)
				{
					block_sum = warp_sums[lane];
				}

			for(std::int32_t offset = warpSize / 2; offset > 0; offset >>= 1)
				{
					block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
				}

			if(lane == 0)
				{
					partial[blockIdx.x] = block_sum;
				}
		}
}

template <Op Kp, typename Tp>
requires core::UnaryOperation<Kp>
__global__ void
assign_sum(
	Tp* __restrict__ x,
	const Tp* __restrict__ sum,
	const std::uint32_t N)
{
	const std::uint32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
	const std::uint32_t stride = blockDim.x * gridDim.x;

	for(std::uint32_t i = tid; i < N; i += stride)
		{
			if constexpr(Kp == Op::SOFTMAX || Kp == Op::SOFTMIN)
				{
					x[i] /= sum[0];
				}
			else if constexpr(Kp == Op::LOGSOFTMAX)
				{
					x[i] = ::cuda::std::log(x[i] / sum[0]);
				}
		}
}

template <Op Kp, ParameterMode Mp, typename T1, typename Dp = core::no_value_t, core::Arithmetic... Is>
requires core::UnaryOperation<Kp>
__global__ void
unary(
	T1* __restrict__ out,
	const std::uint32_t N,
	const Dp            maybe_descriptor,
	Is... param)
{
	const std::uint32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
	const std::uint32_t stride = blockDim.x * gridDim.x;

	for(std::uint32_t i = tid; i < N; i += stride)
		{
			if constexpr(Kp == Op::ABS)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = ::cuda::std::abs(out[i] + noise(maybe_descriptor, i));
						}
					else
						{
							out[i] = ::cuda::std::abs(out[i]);
						}
				}
			else if constexpr(Kp == Op::SIN)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = ::cuda::std::sin(out[i] + noise(maybe_descriptor, i));
						}
					else
						{
							out[i] = ::cuda::std::sin(out[i]);
						}
				}
			else if constexpr(Kp == Op::COS)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = ::cuda::std::cos(out[i] + noise(maybe_descriptor, i));
						}
					else
						{
							out[i] = ::cuda::std::cos(out[i]);
						}
				}
			else if constexpr(Kp == Op::TANH)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = ::cuda::std::tanh(out[i] + noise(maybe_descriptor, i));
						}
					else
						{
							out[i] = ::cuda::std::tanh(out[i]);
						}
				}
			else if constexpr(Kp == Op::NEG)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = -(out[i] + noise(maybe_descriptor, i));
						}
					else
						{
							out[i] = -out[i];
						}
				}
			else if constexpr(Kp == Op::EXP)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = ::cuda::std::exp(out[i] + noise(maybe_descriptor, i));
						}
					else
						{
							out[i] = ::cuda::std::exp(out[i]);
						}
				}
			else if constexpr(Kp == Op::LOG)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = ::cuda::std::log(out[i] + noise(maybe_descriptor, i));
						}
					else
						{
							out[i] = ::cuda::std::log(out[i]);
						}
				}
			else if constexpr(Kp == Op::SQRT)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = ::cuda::std::sqrt(out[i] + noise(maybe_descriptor, i));
						}
					else
						{
							out[i] = ::cuda::std::sqrt(out[i]);
						}
				}
			else if constexpr(Kp == Op::SQUARE)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] *= out[i] + noise(maybe_descriptor, i);
						}
					else
						{
							out[i] *= out[i];
						}
				}
			else if constexpr(Kp == Op::ROUND)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = ::cuda::std::round(out[i] + noise(maybe_descriptor, i));
						}
					else
						{
							out[i] = ::cuda::std::round(out[i]);
						}
				}
			else if constexpr(Kp == Op::SIGMOID)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = 1.0f / (1.0f + ::cuda::std::exp(-(out[i] + noise(maybe_descriptor, i))));
						}
					else
						{
							out[i] = 1.0f / (1.0f + ::cuda::std::exp(-out[i]));
						}
				}
			else if constexpr(Kp == Op::SOFT_RELU)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = ::cuda::std::log(1.0f + ::cuda::std::exp(out[i] + noise(maybe_descriptor, i)));
						}
					else
						{
							out[i] = ::cuda::std::log(1.0f + ::cuda::std::exp(out[i]));
						}
				}
			else if constexpr(Kp == Op::RELU)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							const auto value = out[i] + noise(maybe_descriptor, i);
							out[i]           = value > 0 ? value : 0;
						}
					else
						{
							out[i] = out[i] > 0 ? out[i] : 0;
						}
				}
			else if constexpr(Kp == Op::SOFTMAX || Kp == Op::LOGSOFTMAX)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = ::cuda::std::exp(out[i] + noise(maybe_descriptor, i));
						}
					else
						{
							out[i] = ::cuda::std::exp(out[i]);
						}
				}
			else if constexpr(Kp == Op::SOFTMIN)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = ::cuda::std::exp(-(out[i] + noise(maybe_descriptor, i)));
						}
					else
						{
							out[i] = ::cuda::std::exp(-out[i]);
						}
				}
			else if constexpr(Kp == Op::LEAKY_RELU)
				{
					const float a = static_cast<float>(::cuda::std::get<0>(::cuda::std::tuple{ param... }));

					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							const auto value = out[i] + noise(maybe_descriptor, i);
							out[i]           = value > 0 ? value : (a * value);
						}
					else
						{
							out[i] = out[i] > 0 ? out[i] : (a * out[i]);
						}
				}
			else if constexpr(Kp == Op::ELU)
				{
					const float a = static_cast<float>(::cuda::std::get<0>(::cuda::std::tuple{ param... }));

					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							const auto value = out[i] + noise(maybe_descriptor, i);
							out[i]           = value > 0 ? value : a * (::cuda::std::exp(value) - 1.0f);
						}
					else
						{
							out[i] = out[i] > 0 ? out[i] : a * (::cuda::std::exp(out[i]) - 1.0f);
						}
				}
			else if constexpr(Kp == Op::SWISH)
				{
					const float a = static_cast<float>(::cuda::std::get<0>(::cuda::std::tuple{ param... }));

					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							const auto value = out[i] + noise(maybe_descriptor, i);
							out[i]           = value / (1.0f + ::cuda::std::exp(-a * value));
						}
					else
						{
							out[i] = out[i] / (1.0f + ::cuda::std::exp(-a * out[i]));
						}
				}
			else if constexpr(Kp == Op::LINEAR)
				{
					const float a = static_cast<float>(::cuda::std::get<0>(::cuda::std::tuple{ param... }));
					const float b = static_cast<float>(::cuda::std::get<1>(::cuda::std::tuple{ param... }));

					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = a * (out[i] + noise(maybe_descriptor, i)) + b;
						}
					else
						{
							out[i] = a * out[i] + b;
						}
				}
			else if constexpr(Kp == Op::CLIP)
				{
					const float a = static_cast<float>(::cuda::std::get<0>(::cuda::std::tuple{ param... }));
					const float b = static_cast<float>(::cuda::std::get<1>(::cuda::std::tuple{ param... }));

					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							const ::cuda::std::common_type_t<T1, float> value         = out[i] + noise(maybe_descriptor, i);
							const ::cuda::std::common_type_t<T1, float> upper_clamped = (b < value) ? b : value;
							out[i]                                                    = (upper_clamped < a) ? a : upper_clamped;
						}
					else
						{
							const ::cuda::std::common_type_t<T1, float> upper_clamped = (b < out[i]) ? b : out[i];
							out[i]                                                    = (upper_clamped < a) ? a : upper_clamped;
						}
				}
			else if constexpr(Kp == Op::POW)
				{
					const float a = static_cast<float>(::cuda::std::get<0>(::cuda::std::tuple{ param... }));
					const float b = static_cast<float>(::cuda::std::get<1>(::cuda::std::tuple{ param... }));

					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = a * ::cuda::std::pow(out[i] + noise(maybe_descriptor, i), b);
						}
					else
						{
							out[i] = a * ::cuda::std::pow(out[i], b);
						}
				}
		}
}

}

namespace vext::core::cuda::ops
{

template <Op Kp, ParameterMode Mp, typename T1, core::Arithmetic... Is>
requires core::UnaryOperation<Kp>
void
unary(
	T1*                 out,
	const std::uint32_t N,
	Is... param)
{
	constexpr std::int32_t block_size = 256;
	const std::uint32_t    grid_size  = (N + block_size - 1) / block_size;

	if constexpr(Mp == ParameterMode::PERTURBED)
		{
			const NoiseDescriptor& descriptor = sequentional_noise_descriptor();
			kernel::unary<Kp, Mp><<<grid_size, block_size>>>(out, N, descriptor, param...);
		}
	else
		{
			kernel::unary<Kp, Mp><<<grid_size, block_size>>>(out, N, core::no_value, param...);
		}

	CUDA_CHECK(cudaGetLastError());

	if constexpr(Kp == Op::SOFTMAX || Kp == Op::SOFTMIN || Kp == Op::LOGSOFTMAX)
		{
			T1* d_block_sum = nullptr;
			T1* d_sum       = nullptr;

			CUDA_CHECK(cudaMalloc(&d_block_sum, grid_size * sizeof(T1)));
			CUDA_CHECK(cudaMalloc(&d_sum, sizeof(T1)));

			kernel::reduce_sum<<<grid_size, block_size>>>(d_block_sum, out, N);
			CUDA_CHECK(cudaGetLastError());

			kernel::reduce_sum<<<1, block_size>>>(d_sum, d_block_sum, grid_size);
			CUDA_CHECK(cudaGetLastError());

			kernel::assign_sum<Kp><<<grid_size, block_size>>>(out, d_sum, N);
			CUDA_CHECK(cudaGetLastError());

			CUDA_CHECK(cudaFree(d_block_sum));
			CUDA_CHECK(cudaFree(d_sum));
		}
}

}

#undef CUDA_CHECK

#endif
