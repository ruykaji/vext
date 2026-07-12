#ifndef __VEXT_CORE_CPU_OPERATIONS_CSR_SCATTER_CUH__
#define __VEXT_CORE_CPU_OPERATIONS_CSR_SCATTER_CUH__

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

template <CSRScatterOperation Kp, typename T1, typename T2>
__global__ void
csr_scatter(
	T1* __restrict__ out,
	const T2* __restrict__ src,
	const std::uint32_t* __restrict__ head,
	const std::uint32_t* __restrict__ tail,
	const std::uint32_t N,
	const std::uint32_t S)
{
	const std::uint32_t lane       = threadIdx.x % warpSize;
	const std::uint32_t warp_id    = threadIdx.x / warpSize;
	const std::uint32_t warp_count = blockDim.x / warpSize;

	for(std::uint32_t i = blockIdx.x; i < N; i += gridDim.x)
		{
			const std::uint32_t start = head[i];
			const std::uint32_t end   = head[i + 1];

			if(start == end)
				{
					continue;
				}

			const float scale = 1.0f / (end - start);

			for(std::uint32_t k = warp_id; k < S; k += warp_count)
				{
					T1 accumulator = 0;

					if constexpr(Kp == CSRScatterOperation::PROD)
						{
							accumulator = 1;
						}
					else if constexpr(Kp == CSRScatterOperation::MIN)
						{
							accumulator = ::cuda::std::numeric_limits<T1>::max();
						}
					else if constexpr(Kp == CSRScatterOperation::MAX)
						{
							accumulator = ::cuda::std::numeric_limits<T1>::lowest();
						}

					for(std::uint32_t h = start + lane; h < end; h += warpSize)
						{
							const std::uint32_t index = tail[h] * S + k;

							if constexpr(Kp == CSRScatterOperation::PROD)
								{
									accumulator *= src[index];
								}
							else if constexpr(Kp == CSRScatterOperation::MIN)
								{
									accumulator = ::cuda::std::min<T1>(accumulator, src[index]);
								}
							else if constexpr(Kp == CSRScatterOperation::MAX)
								{
									accumulator = ::cuda::std::max<T1>(accumulator, src[index]);
								}
							else if constexpr(Kp == CSRScatterOperation::VAR || Kp == CSRScatterOperation::STD)
								{
									const float diff = src[index] - out[i * S + k];
									accumulator += diff * diff;
								}
							else
								{
									accumulator += src[index];
								}
						}

					for(std::uint32_t offset = warpSize / 2; offset > 0; offset >>= 1)
						{
							const T2 shuffled = __shfl_down_sync(0xffffffff, accumulator, offset);

							if constexpr(Kp == CSRScatterOperation::PROD)
								{
									accumulator *= shuffled;
								}
							else if constexpr(Kp == CSRScatterOperation::MIN)
								{
									accumulator = (shuffled < accumulator) ? shuffled : accumulator;
								}
							else if constexpr(Kp == CSRScatterOperation::MAX)
								{
									accumulator = (shuffled > accumulator) ? shuffled : accumulator;
								}
							else
								{
									accumulator += shuffled;
								}
						}

					if(lane == 0)
						{
							if constexpr(Kp == CSRScatterOperation::MEAN || Kp == CSRScatterOperation::VAR)
								{
									out[i * S + k] = accumulator * scale;
								}
							else if constexpr(Kp == CSRScatterOperation::STD)
								{
									out[i * S + k] = ::cuda::std::sqrt(accumulator * scale);
								}
							else
								{
									out[i * S + k] = accumulator;
								}
						}
				}
		}
}

}

namespace vext::core::cuda::operations
{

template <CSRScatterOperation Kp, typename T1, typename T2>
void
csr_scatter(
	T1*                  out,
	const T2*            src,
	const std::uint32_t* head,
	const std::uint32_t* tail,
	const std::uint32_t  N,
	const std::uint32_t  S)
{
	constexpr std::uint32_t block_size = 256;
	const std::uint32_t     grid_size  = (N + block_size - 1) / block_size;

	if constexpr(Kp == CSRScatterOperation::VAR || Kp == CSRScatterOperation::STD)
		{
			kernel::csr_scatter<CSRScatterOperation::MEAN, T1, T2><<<grid_size, block_size>>>(out, src, head, tail, N, S);
			CUDA_CHECK(cudaGetLastError());

			kernel::csr_scatter<Kp, T1, T2><<<grid_size, block_size>>>(out, src, head, tail, N, S);
			CUDA_CHECK(cudaGetLastError());
		}
	else
		{
			kernel::csr_scatter<Kp, T1, T2><<<grid_size, block_size>>>(out, src, head, tail, N, S);
			CUDA_CHECK(cudaGetLastError());
		}
}

}

#undef CUDA_CHECK

#endif
