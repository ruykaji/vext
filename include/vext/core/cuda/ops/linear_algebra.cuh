#ifndef __VEXT_CORE_CUDA_OPERATIONS_LINEAR_ALGEBRA_CUH__
#define __VEXT_CORE_CUDA_OPERATIONS_LINEAR_ALGEBRA_CUH__

#include <iostream>

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

template <ParameterMode Mp, typename Dp = core::no_value_t, typename T1, typename T2, typename T3>
__global__ void
matmul(
	T1* __restrict__ out,
	const T2* __restrict__ a,
	const T3* __restrict__ b,
	const std::uint32_t M,
	const std::uint32_t P,
	const std::uint32_t N,
	const Dp            maybe_descriptor)
{
	__shared__ T2                                    a_tile[16][16];
	__shared__ ::cuda::std::common_type_t<T3, float> b_tile[16][16];

	const std::uint32_t row        = blockIdx.y * 16 + threadIdx.y;
	const std::uint32_t col        = blockIdx.x * 16 + threadIdx.x;
	const std::uint32_t tile_count = (P + 16 - 1) / 16;

	::cuda::std::common_type_t<T2, T3, float> sum = 0;

	for(std::uint32_t tile = 0; tile < tile_count; ++tile)
		{
			const std::uint32_t a_col = tile * 16 + threadIdx.x;
			const std::uint32_t b_row = tile * 16 + threadIdx.y;

			a_tile[threadIdx.y][threadIdx.x] = (row < M && a_col < P) ? a[row * P + a_col] : 0;

			if(b_row < P && col < N)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							const std::uint32_t index        = b_row * N + col;
							b_tile[threadIdx.y][threadIdx.x] = b[index] + noise(maybe_descriptor, index);
						}
					else
						{
							b_tile[threadIdx.y][threadIdx.x] = b[b_row * N + col];
						}
				}
			else
				{
					b_tile[threadIdx.y][threadIdx.x] = 0;
				}

			__syncthreads();

			#pragma unroll
			for(std::int32_t k = 0; k < 16; ++k)
				{
					sum += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
				}

			__syncthreads();
		}

	if(row < M && col < N)
		{
			const std::size_t index = static_cast<std::size_t>(row) * N + col;
			out[index]              = sum;
		}
}

}

namespace vext::core::cuda::ops
{

template <ParameterMode Mp = ParameterMode::PLAIN, typename T1, typename T2, typename T3>
void
matmul(
	T1*                 out,
	const T2*           a,
	const T3*           b,
	const std::uint32_t M,
	const std::uint32_t P,
	const std::uint32_t N)
{
	const dim3 block(16, 16);
	const dim3 grid((N + 16 - 1) / 16, (M + 16 - 1) / 16);

	if constexpr(Mp == ParameterMode::PERTURBED)
		{
			kernel::matmul<Mp><<<grid, block>>>(out, a, b, M, P, N, sequentional_noise_descriptor());
		}
	else
		{
			kernel::matmul<Mp><<<grid, block>>>(out, a, b, M, P, N, core::no_value);
		}

	CUDA_CHECK(cudaGetLastError());
}

}

#undef CUDA_CHECK

#endif
