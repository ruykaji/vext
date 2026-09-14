#ifndef __VEXT_CORE_CPU_OPS_CSR_SCATTER_CUH__
#define __VEXT_CORE_CPU_OPS_CSR_SCATTER_CUH__

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

template <Op Kp, ParameterMode Mp, typename T1, typename T2, typename Dp = core::no_value_t>
requires core::SparseReductionOperation<Kp>
__global__ void
csr_scatter(
	T1* __restrict__ out,
	const T2* __restrict__ src,
	const std::uint32_t* __restrict__ head,
	const std::uint32_t* __restrict__ tail,
	const std::uint32_t N,
	const std::uint32_t S,
	const Dp            maybe_descriptor)
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

					for(std::uint32_t h = start + lane; h < end; h += warpSize)
						{
							if constexpr(Kp == Op::PROD)
								{
									if constexpr(Mp == ParameterMode::PERTURBED)
										{
											const std::uint32_t index = tail[h] * S + k;
											accumulator *= src[index] + noise(maybe_descriptor, index);
										}
									else
										{
											accumulator *= src[tail[h] * S + k];
										}
								}
							else if constexpr(Kp == Op::MIN)
								{
									if constexpr(Mp == ParameterMode::PERTURBED)
										{
											const std::uint32_t index = tail[h] * S + k;
											accumulator               = ::cuda::std::min<T1>(accumulator, src[index] + noise(maybe_descriptor, index));
										}
									else
										{
											accumulator = ::cuda::std::min<T1>(accumulator, src[tail[h] * S + k]);
										}
								}
							else if constexpr(Kp == Op::MAX)
								{
									if constexpr(Mp == ParameterMode::PERTURBED)
										{
											const std::uint32_t index = tail[h] * S + k;
											accumulator               = ::cuda::std::max<T1>(accumulator, src[index] + noise(maybe_descriptor, index));
										}
									else
										{
											accumulator = ::cuda::std::max<T1>(accumulator, src[tail[h] * S + k]);
										}
								}
							else if constexpr(Kp == Op::VAR || Kp == Op::STD)
								{
									float diff = 0.0f;

									if constexpr(Mp == ParameterMode::PERTURBED)
										{
											const std::uint32_t index = tail[h] * S + k;
											diff                      = (src[index] + noise(maybe_descriptor, index)) - out[i * S + k];
										}
									else
										{
											diff = src[tail[h] * S + k] - out[i * S + k];
										}

									accumulator += diff * diff;
								}
							else
								{
									if constexpr(Mp == ParameterMode::PERTURBED)
										{
											const std::uint32_t index = tail[h] * S + k;
											accumulator += src[index] + noise(maybe_descriptor, index);
										}
									else
										{
											accumulator += src[tail[h] * S + k];
										}
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

					if(lane == 0)
						{
							if constexpr(Kp == Op::MEAN || Kp == Op::VAR)
								{
									out[i * S + k] = accumulator * scale;
								}
							else if constexpr(Kp == Op::STD)
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

namespace vext::core::cuda::ops
{

template <Op Kp, ParameterMode Mp, typename T1, typename T2>
requires core::SparseReductionOperation<Kp>
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

	if constexpr(Kp == Op::VAR || Kp == Op::STD)
		{
			if constexpr(Mp == ParameterMode::PERTURBED)
				{
					const NoiseDescriptor& descriptor = sequentional_noise_descriptor();
					kernel::csr_scatter<Op::MEAN, Mp><<<grid_size, block_size>>>(out, src, head, tail, N, S, descriptor);
					CUDA_CHECK(cudaGetLastError());

					kernel::csr_scatter<Kp, Mp><<<grid_size, block_size>>>(out, src, head, tail, N, S, descriptor);
					CUDA_CHECK(cudaGetLastError());
				}
			else
				{
					kernel::csr_scatter<Op::MEAN, Mp><<<grid_size, block_size>>>(out, src, head, tail, N, S, core::no_value);
					CUDA_CHECK(cudaGetLastError());

					kernel::csr_scatter<Kp, Mp><<<grid_size, block_size>>>(out, src, head, tail, N, S, core::no_value);
					CUDA_CHECK(cudaGetLastError());
				}
		}
	else
		{
			if constexpr(Mp == ParameterMode::PERTURBED)
				{
					kernel::csr_scatter<Kp, Mp><<<grid_size, block_size>>>(out, src, head, tail, N, S, sequentional_noise_descriptor());
				}
			else
				{
					kernel::csr_scatter<Kp, Mp><<<grid_size, block_size>>>(out, src, head, tail, N, S, core::no_value);
				}

			CUDA_CHECK(cudaGetLastError());
		}
}

}

#undef CUDA_CHECK

#endif
