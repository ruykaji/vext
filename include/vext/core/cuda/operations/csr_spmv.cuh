#ifndef __VEXT_CORE_CPU_OPERATIONS_CSR_SPMV_CUH__
#define __VEXT_CORE_CPU_OPERATIONS_CSR_SPMV_CUH__

#include <iostream>

#include <cuda/std/algorithm>
#include <cuda/std/cmath>
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

template <CSRSpMVOperation Kp, typename T1, typename T2, typename T3>
__global__ void
csr_spmv(
	T1* __restrict__ y,
	const T2* __restrict__ A,
	const std::uint32_t* __restrict__ head,
	const std::uint32_t* __restrict__ tail,
	const T3* __restrict__ x,
	const std::uint32_t N)
{
	for(std::uint32_t i = blockIdx.x; i < N; i += gridDim.x)
		{
			const std::uint32_t start = head[i];
			const std::uint32_t end   = head[i + 1];

			T1 accumulator = 0;

			if constexpr(Kp == CSRSpMVOperation::PROD)
				{
					accumulator = 1;
				}
			else if constexpr(Kp == CSRSpMVOperation::MIN)
				{
					accumulator = ::cuda::std::numeric_limits<T1>::max();
				}
			else if constexpr(Kp == CSRSpMVOperation::MAX)
				{
					accumulator = ::cuda::std::numeric_limits<T1>::lowest();
				}

			for(std::uint32_t h = start + threadIdx.x; h < end; h += blockDim.x)
				{
					const T1 prod = A[h] * x[tail[h]];

					if constexpr(Kp == CSRSpMVOperation::PROD)
						{
							accumulator *= prod;
						}
					else if constexpr(Kp == CSRSpMVOperation::MIN)
						{
							accumulator = ::cuda::std::min<T1>(accumulator, prod);
						}
					else if constexpr(Kp == CSRSpMVOperation::MAX)
						{
							accumulator = ::cuda::std::max<T1>(accumulator, prod);
						}
					else if constexpr(Kp == CSRSpMVOperation::VAR || Kp == CSRSpMVOperation::STD)
						{
							const float diff = prod - y[i];
							accumulator += diff * diff;
						}
					else
						{
							accumulator += prod;
						}
				}

			for(std::uint32_t offset = warpSize / 2; offset > 0; offset >>= 1)
				{
					const T2 shuffled = __shfl_down_sync(0xffffffff, accumulator, offset);

					if constexpr(Kp == CSRSpMVOperation::PROD)
						{
							accumulator *= shuffled;
						}
					else if constexpr(Kp == CSRSpMVOperation::MIN)
						{
							accumulator = (shuffled < accumulator) ? shuffled : accumulator;
						}
					else if constexpr(Kp == CSRSpMVOperation::MAX)
						{
							accumulator = (shuffled > accumulator) ? shuffled : accumulator;
						}
					else
						{
							accumulator += shuffled;
						}
				}

			__shared__ T2 warp_accumulate[32];

			const std::uint32_t lane    = threadIdx.x % warpSize;
			const std::uint32_t warp_id = threadIdx.x / warpSize;

			if(lane == 0)
				{
					warp_accumulate[warp_id] = accumulator;
				}

			__syncthreads();

			T2 block_accumulate = 0;

			if constexpr(Kp == CSRSpMVOperation::PROD)
				{
					block_accumulate = 1;
				}
			else if constexpr(Kp == CSRSpMVOperation::MIN)
				{
					block_accumulate = ::cuda::std::numeric_limits<T1>::max();
				}
			else if constexpr(Kp == CSRSpMVOperation::MAX)
				{
					block_accumulate = ::cuda::std::numeric_limits<T1>::lowest();
				}

			if(warp_id == 0)
				{
					const std::uint32_t warp_count = (blockDim.x + warpSize - 1) / warpSize;

					if(lane < warp_count)
						{
							block_accumulate = warp_accumulate[lane];
						}

					for(std::uint32_t offset = warpSize / 2; offset > 0; offset >>= 1)
						{
							const T2 shuffled = __shfl_down_sync(0xffffffff, block_accumulate, offset);

							if constexpr(Kp == CSRSpMVOperation::PROD)
								{
									block_accumulate *= shuffled;
								}
							else if constexpr(Kp == CSRSpMVOperation::MIN)
								{
									block_accumulate = (shuffled < block_accumulate) ? shuffled : block_accumulate;
								}
							else if constexpr(Kp == CSRSpMVOperation::MAX)
								{
									block_accumulate = (shuffled > block_accumulate) ? shuffled : block_accumulate;
								}
							else
								{
									block_accumulate += shuffled;
								}
						}

					if(lane == 0)
						{
							if constexpr(Kp == CSRSpMVOperation::MEAN || Kp == CSRSpMVOperation::VAR)
								{
									const float scale = 1.0f / (end - start);
									y[i]              = block_accumulate * scale;
								}
							else if constexpr(Kp == CSRSpMVOperation::STD)
								{
									const float scale = 1.0f / (end - start);
									y[i]              = ::cuda::std::sqrt(block_accumulate * scale);
								}
							else
								{
									y[i] = block_accumulate;
								}
						}
				}

			__syncthreads();
		}
}

}

namespace vext::core::cuda::operations
{

template <CSRSpMVOperation Kp, typename T1, typename T2, typename T3>
void
csr_spmv(
	T1*                  y,
	const T2*            A,
	const std::uint32_t* head,
	const std::uint32_t* tail,
	const T3*            x,
	const std::uint32_t  N)
{
	constexpr std::uint32_t block_size = 256;
	const std::uint32_t     grid_size  = (N + block_size - 1) / block_size;

	if constexpr(Kp == CSRSpMVOperation::VAR || Kp == CSRSpMVOperation::STD)
		{
			kernel::csr_spmv<CSRSpMVOperation::MEAN, T1, T2><<<grid_size, block_size>>>(y, A, head, tail, x, N);
			CUDA_CHECK(cudaGetLastError());

			kernel::csr_spmv<Kp, T1, T2, T3><<<grid_size, block_size>>>(y, A, head, tail, x, N);
			CUDA_CHECK(cudaGetLastError());
		}
	else
		{
			kernel::csr_spmv<Kp, T1, T2, T3><<<grid_size, block_size>>>(y, A, head, tail, x, N);
			CUDA_CHECK(cudaGetLastError());
		}
}

}

#undef CUDA_CHECK

#endif
