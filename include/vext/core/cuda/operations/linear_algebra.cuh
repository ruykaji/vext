#ifndef __VEXT_CORE_CUDA_OPERATIONS_LINEAR_ALGEBRA_CUH__
#define __VEXT_CORE_CUDA_OPERATIONS_LINEAR_ALGEBRA_CUH__

#include <iostream>

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

template <typename T1, typename T2>
__global__ void
matmul(
	std::common_type_t<T1, T2>* __restrict__ out,
	const T1* __restrict__ a,
	const T2* __restrict__ b,
	const std::uint32_t M,
	const std::uint32_t P,
	const std::uint32_t N)
{
	__shared__ T1 a_tile[16][16];
	__shared__ T2 b_tile[16][16];

	const std::uint32_t row        = blockIdx.y * 16 + threadIdx.y;
	const std::uint32_t col        = blockIdx.x * 16 + threadIdx.x;
	const std::uint32_t tile_count = (P + 16 - 1) / 16;

	std::common_type_t<T1, T2> sum = 0;

	for(std::uint32_t tile = 0; tile < tile_count; ++tile)
		{
			const std::uint32_t a_col = tile * 16 + threadIdx.x;
			const std::uint32_t b_row = tile * 16 + threadIdx.y;

			a_tile[threadIdx.y][threadIdx.x] = (row < M && a_col < P) ? a[row * P + a_col] : 0;
			b_tile[threadIdx.y][threadIdx.x] = (b_row < P && col < N) ? b[b_row * N + col] : 0;

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
			out[static_cast<std::size_t>(row) * N + col] = sum;
		}
}

}

namespace vext::core::cuda::operations
{

template <typename T1, typename T2>
void
matmul(
	std::common_type_t<T1, T2>* out,
	const T1*                   a,
	const T2*                   b,
	const std::uint32_t         M,
	const std::uint32_t         P,
	const std::uint32_t         N)
{
	const dim3 block(16, 16);
	const dim3 grid((N + 16 - 1) / 16, (M + 16 - 1) / 16);

	kernel::matmul<T1, T2><<<grid, block>>>(out, a, b, M, P, N);
	CUDA_CHECK(cudaGetLastError());
}

}

#undef CUDA_CHECK

#endif
