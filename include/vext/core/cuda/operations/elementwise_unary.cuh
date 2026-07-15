#ifndef __VEXT_CORE_CUDA_OPERATIONS_ELEMENTWISE_UNARY_CUH__
#define __VEXT_CORE_CUDA_OPERATIONS_ELEMENTWISE_UNARY_CUH__

#include <iostream>

#include <cuda/std/algorithm>
#include <cuda/std/cmath>
#include <cuda/std/tuple>
#include <cuda_runtime.h>

#include <vext/core/type.hpp>

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

namespace vext::core::cuda::operations::kernel
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

template <UnaryOperation Kp, typename Tp>
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
			if constexpr(Kp == UnaryOperation::SOFTMAX || Kp == UnaryOperation::SOFTMIN)
				{
					x[i] /= sum[0];
				}
			else if constexpr(Kp == UnaryOperation::LOGSOFTMAX)
				{
					x[i] = ::cuda::std::log(x[i] / sum[0]);
				}
		}
}

template <UnaryOperation Kp, typename T1, core::Arithmetic... Is>
__global__ void
unary(
	T1* __restrict__ out,
	const std::uint32_t N,
	Is... param)
{
	const std::uint32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
	const std::uint32_t stride = blockDim.x * gridDim.x;

	for(std::uint32_t i = tid; i < N; i += stride)
		{
			if constexpr(Kp == UnaryOperation::ABS)
				{
					out[i] = ::cuda::std::abs(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::SIN)
				{
					out[i] = ::cuda::std::sin(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::COS)
				{
					out[i] = ::cuda::std::cos(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::TANH)
				{
					out[i] = ::cuda::std::tanh(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::NEG)
				{
					out[i] = -out[i];
				}
			else if constexpr(Kp == UnaryOperation::EXP)
				{
					out[i] = ::cuda::std::exp(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::LOG)
				{
					out[i] = ::cuda::std::log(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::SQRT)
				{
					out[i] = ::cuda::std::sqrt(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::SQUARE)
				{
					out[i] *= out[i];
				}
			else if constexpr(Kp == UnaryOperation::ROUND)
				{
					out[i] = ::cuda::std::round(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::SIGMOID)
				{
					out[i] = 1.0f / (1.0f + ::cuda::std::exp(-out[i]));
				}
			else if constexpr(Kp == UnaryOperation::SOFT_RELU)
				{
					out[i] = ::cuda::std::log(1.0f + ::cuda::std::exp(out[i]));
				}
			else if constexpr(Kp == UnaryOperation::RELU)
				{
					out[i] = out[i] > 0 ? out[i] : 0;
				}
			else if constexpr(Kp == UnaryOperation::SOFTMAX || Kp == UnaryOperation::LOGSOFTMAX)
				{
					out[i] = ::cuda::std::exp(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::SOFTMIN)
				{
					out[i] = ::cuda::std::exp(-out[i]);
				}
			else if constexpr(Kp == UnaryOperation::LEAKY_RELU)
				{
					const float a = static_cast<float>(::cuda::std::get<0>(::cuda::std::tuple{ param... }));
					out[i]        = out[i] > 0 ? out[i] : (a * out[i]);
				}
			else if constexpr(Kp == UnaryOperation::ELU)
				{
					const float a = static_cast<float>(::cuda::std::get<0>(::cuda::std::tuple{ param... }));
					out[i]        = out[i] > 0 ? out[i] : a * (::cuda::std::exp(out[i]) - 1.0f);
				}
			else if constexpr(Kp == UnaryOperation::SWISH)
				{
					const float a = static_cast<float>(::cuda::std::get<0>(::cuda::std::tuple{ param... }));
					out[i]        = out[i] / (1.0f + ::cuda::std::exp(-a * out[i]));
				}
			else if constexpr(Kp == UnaryOperation::LINEAR)
				{
					const float a = static_cast<float>(::cuda::std::get<0>(::cuda::std::tuple{ param... }));
					const float b = static_cast<float>(::cuda::std::get<1>(::cuda::std::tuple{ param... }));
					out[i]        = a * out[i] + b;
				}
			else if constexpr(Kp == UnaryOperation::CLIP)
				{
					const float a = static_cast<float>(::cuda::std::get<0>(::cuda::std::tuple{ param... }));
					const float b = static_cast<float>(::cuda::std::get<1>(::cuda::std::tuple{ param... }));
					out[i]        = ::cuda::std::max(a, ::cuda::std::min(b, out[i]));
				}
			else if constexpr(Kp == UnaryOperation::POW)
				{
					const float a = static_cast<float>(::cuda::std::get<0>(::cuda::std::tuple{ param... }));
					const float b = static_cast<float>(::cuda::std::get<1>(::cuda::std::tuple{ param... }));
					out[i]        = a * ::cuda::std::pow(out[i], b);
				}
		}
}

}

namespace vext::core::cuda::operations
{

template <UnaryOperation Kp, typename T1, core::Arithmetic... Is>
void
unary(
	T1*                 out,
	const std::uint32_t N,
	Is... param)
{
	constexpr std::int32_t block_size = 256;
	const std::uint32_t    grid_size  = (N + block_size - 1) / block_size;

	if constexpr(Kp == UnaryOperation::SOFTMAX || Kp == UnaryOperation::SOFTMIN || Kp == UnaryOperation::LOGSOFTMAX)
		{
			kernel::unary<Kp, T1><<<grid_size, block_size>>>(out, N, param...);
			CUDA_CHECK(cudaGetLastError());

			T1* d_block_sum = nullptr;
			T1* d_sum       = nullptr;

			CUDA_CHECK(cudaMalloc(&d_block_sum, grid_size * sizeof(T1)));
			CUDA_CHECK(cudaMalloc(&d_sum, sizeof(T1)));

			kernel::reduce_sum<T1><<<grid_size, block_size>>>(d_block_sum, out, N);
			CUDA_CHECK(cudaGetLastError());

			kernel::reduce_sum<T1><<<1, block_size>>>(d_sum, d_block_sum, grid_size);
			CUDA_CHECK(cudaGetLastError());

			kernel::assign_sum<Kp, T1><<<grid_size, block_size>>>(out, d_sum, N);
			CUDA_CHECK(cudaGetLastError());

			CUDA_CHECK(cudaFree(d_block_sum));
			CUDA_CHECK(cudaFree(d_sum));
		}
	else
		{
			kernel::unary<Kp, T1><<<grid_size, block_size>>>(out, N, param...);
			CUDA_CHECK(cudaGetLastError());
		}
}

}

#undef CUDA_CHECK

#endif
