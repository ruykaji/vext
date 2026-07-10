#ifndef __VEXT_CORE_CUDA_OPERATIONS_REDUCTION_CUH__
#define __VEXT_CORE_CUDA_OPERATIONS_REDUCTION_CUH__

#include <iostream>
#include <vector>

#include <cuda/std/algorithm>
#include <cuda/std/cmath>
#include <cuda/std/limits>
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

struct ReductionMeta
{
	std::uint32_t dims[MAX_RANK];
	std::uint32_t strides[MAX_RANK];
	std::uint32_t size = 0;
};

// clang-format off
__forceinline__  __device__ std::uint32_t
axis_offset(
	std::uint32_t        element, 
    const ReductionMeta& meta)
{
	std::uint32_t offset = 0;

	for(std::int32_t i = meta.size - 1; i >=0 ; --i)
		{
            const std::uint32_t d     = meta.dims[i];
			const std::uint32_t q     = element / meta.dims[i];
            const std::uint32_t index = element - q * d;

			element = q;
			offset += index * meta.strides[i];
		}

	return offset;
}

// clang-format on

template <ReductionOperation Kp, typename T1, typename T2>
__global__ void
reduce(
	T1* __restrict__ out,
	const T2* __restrict__ src,
	const std::uint32_t N,
	const std::uint32_t M,
	const ReductionMeta keep_meta,
	const ReductionMeta reduce_meta)
{
	for(std::uint32_t i = blockIdx.x; i < N; i += gridDim.x)
		{
			T1 accumulator = 0;

			if constexpr(Kp == ReductionOperation::PROD)
				{
					accumulator = 1;
				}
			else if constexpr(Kp == ReductionOperation::MIN)
				{
					accumulator = ::cuda::std::numeric_limits<T1>::max();
				}
			else if constexpr(Kp == ReductionOperation::MAX)
				{
					accumulator = ::cuda::std::numeric_limits<T1>::lowest();
				}

			const std::uint32_t keep_offset = axis_offset(i, keep_meta);

			for(std::uint32_t j = threadIdx.x; j < M; j += blockDim.x)
				{
					const std::uint64_t reduce_offset = axis_offset(j, reduce_meta);

					if constexpr(Kp == ReductionOperation::PROD)
						{
							accumulator *= static_cast<T1>(src[keep_offset + reduce_offset]);
						}
					else if constexpr(Kp == ReductionOperation::MIN)
						{
							accumulator = ::cuda::std::min(accumulator, static_cast<T1>(src[keep_offset + reduce_offset]));
						}
					else if constexpr(Kp == ReductionOperation::MAX)
						{
							accumulator = ::cuda::std::max(accumulator, static_cast<T1>(src[keep_offset + reduce_offset]));
						}
					else if constexpr(Kp == ReductionOperation::VAR || Kp == ReductionOperation::STD)
						{
							accumulator += ::cuda::std::pow(src[keep_offset + reduce_offset] - out[i], 2.0f);
						}
					else
						{
							accumulator += static_cast<T1>(src[keep_offset + reduce_offset]);
						}
				}

			for(std::uint32_t offset = warpSize / 2; offset > 0; offset >>= 1)
				{
					const T2 shuffled = __shfl_down_sync(0xffffffff, accumulator, offset);

					if constexpr(Kp == ReductionOperation::PROD)
						{
							accumulator *= shuffled;
						}
					else if constexpr(Kp == ReductionOperation::MIN)
						{
							accumulator = (shuffled < accumulator) ? shuffled : accumulator;
						}
					else if constexpr(Kp == ReductionOperation::MAX)
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

			if constexpr(Kp == ReductionOperation::PROD)
				{
					block_accumulate = 1;
				}
			else if constexpr(Kp == ReductionOperation::MIN)
				{
					block_accumulate = ::cuda::std::numeric_limits<T1>::max();
				}
			else if constexpr(Kp == ReductionOperation::MAX)
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

							if constexpr(Kp == ReductionOperation::PROD)
								{
									block_accumulate *= shuffled;
								}
							else if constexpr(Kp == ReductionOperation::MIN)
								{
									block_accumulate = (shuffled < block_accumulate) ? shuffled : block_accumulate;
								}
							else if constexpr(Kp == ReductionOperation::MAX)
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
							if constexpr(Kp == ReductionOperation::MEAN || Kp == ReductionOperation::VAR)
								{
									out[i] = block_accumulate / M;
								}
							else if constexpr(Kp == ReductionOperation::STD)
								{
									out[i] = ::cuda::std::sqrt(block_accumulate / M);
								}
							else
								{
									out[i] = block_accumulate;
								}
						}
				}

			__syncthreads();
		}
}

}

namespace vext::core::cuda::operations
{

template <ReductionOperation Kp, typename T1, typename T2>
void
reduce(
	T1*                               out,
	const T2*                         src,
	const std::uint32_t               N,
	const std::uint32_t               M,
	const std::vector<std::uint32_t>& keep_dims,
	const std::vector<std::uint32_t>& keep_strides,
	const std::vector<std::uint32_t>& reduce_dims,
	const std::vector<std::uint32_t>& reduce_strides)
{
	constexpr std::uint32_t block_size = 256;
	const std::uint32_t     grid_size  = (N + block_size - 1) / block_size;

	kernel::ReductionMeta keep_meta = { .size = static_cast<std::uint32_t>(keep_dims.size()) };

	for(std::uint32_t i = 0; i < keep_meta.size; ++i)
		{
			keep_meta.dims[i]    = keep_dims[i];
			keep_meta.strides[i] = keep_strides[i];
		}

	kernel::ReductionMeta reduce_meta = { .size = static_cast<std::uint32_t>(reduce_dims.size()) };

	for(std::uint32_t i = 0; i < reduce_meta.size; ++i)
		{
			reduce_meta.dims[i]    = reduce_dims[i];
			reduce_meta.strides[i] = reduce_strides[i];
		}

	if constexpr(Kp == ReductionOperation::VAR || Kp == ReductionOperation::STD)
		{
			kernel::reduce<ReductionOperation::MEAN, T1, T2><<<grid_size, block_size>>>(out, src, N, M, keep_meta, reduce_meta);
			CUDA_CHECK(cudaGetLastError());

			kernel::reduce<Kp, T1, T2><<<grid_size, block_size>>>(out, src, N, M, keep_meta, reduce_meta);
			CUDA_CHECK(cudaGetLastError());
		}
	else
		{
			kernel::reduce<Kp, T1, T2><<<grid_size, block_size>>>(out, src, N, M, keep_meta, reduce_meta);
			CUDA_CHECK(cudaGetLastError());
		}
}

}

#undef CUDA_CHECK

#endif
