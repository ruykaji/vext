#ifndef __VEXT_CORE_CPU_OPS_CSR_SPMV_CUH__
#define __VEXT_CORE_CPU_OPS_CSR_SPMV_CUH__

#include <iostream>

#include <cuda/std/algorithm>
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

template <Op Kp, ParameterMode Mp, typename T1, typename T2, typename T3, typename Dp = core::no_value_t>
requires core::SparseReductionOperation<Kp>
__global__ void
csr_spmv(
	T1* __restrict__ y,
	const T2* __restrict__ A,
	const std::uint32_t* __restrict__ head,
	const std::uint32_t* __restrict__ tail,
	const T3* __restrict__ x,
	const std::uint32_t N,
	const Dp            maybe_descriptor)
{
	for(std::uint32_t i = blockIdx.x; i < N; i += gridDim.x)
		{
			const std::uint32_t start = head[i];
			const std::uint32_t end   = head[i + 1];

			T1 accumulator = 0;

			if constexpr(Kp == Op::PROD)
				{
					accumulator = 1;
				}
			else if constexpr(Kp == Op::MIN)
				{
					accumulator = ::cuda::std::numeric_limits<T1>::max();
				}
			else if constexpr(Kp == Op::MAX)
				{
					accumulator = ::cuda::std::numeric_limits<T1>::lowest();
				}

			for(std::uint32_t h = start + threadIdx.x; h < end; h += blockDim.x)
				{
					T1 prod = 0;

					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							prod = (A[h] + noise(maybe_descriptor, h)) * x[tail[h]];
						}
					else
						{
							prod = A[h] * x[tail[h]];
						}

					if constexpr(Kp == Op::PROD)
						{
							accumulator *= prod;
						}
					else if constexpr(Kp == Op::MIN)
						{
							accumulator = ::cuda::std::min<T1>(accumulator, prod);
						}
					else if constexpr(Kp == Op::MAX)
						{
							accumulator = ::cuda::std::max<T1>(accumulator, prod);
						}
					else if constexpr(Kp == Op::VAR || Kp == Op::STD)
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

					if constexpr(Kp == Op::PROD)
						{
							accumulator *= shuffled;
						}
					else if constexpr(Kp == Op::MIN)
						{
							accumulator = (shuffled < accumulator) ? shuffled : accumulator;
						}
					else if constexpr(Kp == Op::MAX)
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

			if constexpr(Kp == Op::PROD)
				{
					block_accumulate = 1;
				}
			else if constexpr(Kp == Op::MIN)
				{
					block_accumulate = ::cuda::std::numeric_limits<T1>::max();
				}
			else if constexpr(Kp == Op::MAX)
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

							if constexpr(Kp == Op::PROD)
								{
									block_accumulate *= shuffled;
								}
							else if constexpr(Kp == Op::MIN)
								{
									block_accumulate = (shuffled < block_accumulate) ? shuffled : block_accumulate;
								}
							else if constexpr(Kp == Op::MAX)
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
							if constexpr(Kp == Op::MEAN || Kp == Op::VAR)
								{
									const float scale = 1.0f / (end - start);
									y[i]              = block_accumulate * scale;
								}
							else if constexpr(Kp == Op::STD)
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

namespace vext::core::cuda::ops
{

template <Op Kp, ParameterMode Mp, typename T1, typename T2, typename T3>
requires core::SparseReductionOperation<Kp>
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

	if constexpr(Kp == Op::VAR || Kp == Op::STD)
		{
			if constexpr(Mp == ParameterMode::PERTURBED)
				{
					const NoiseDescriptor& descriptor = sequentional_noise_descriptor();
					kernel::csr_spmv<Op::MEAN, Mp><<<grid_size, block_size>>>(y, A, head, tail, x, N, descriptor);
					CUDA_CHECK(cudaGetLastError());

					kernel::csr_spmv<Kp, Mp><<<grid_size, block_size>>>(y, A, head, tail, x, N, descriptor);
					CUDA_CHECK(cudaGetLastError());
				}
			else
				{
					kernel::csr_spmv<Op::MEAN, Mp><<<grid_size, block_size>>>(y, A, head, tail, x, N, core::no_value);
					CUDA_CHECK(cudaGetLastError());

					kernel::csr_spmv<Kp, Mp><<<grid_size, block_size>>>(y, A, head, tail, x, N, core::no_value);
					CUDA_CHECK(cudaGetLastError());
				}
		}
	else
		{
			if constexpr(Mp == ParameterMode::PERTURBED)
				{
					kernel::csr_spmv<Kp, Mp><<<grid_size, block_size>>>(y, A, head, tail, x, N, sequentional_noise_descriptor());
				}
			else
				{
					kernel::csr_spmv<Kp, Mp><<<grid_size, block_size>>>(y, A, head, tail, x, N, core::no_value);
				}

			CUDA_CHECK(cudaGetLastError());
		}
}

}

#undef CUDA_CHECK

#endif
