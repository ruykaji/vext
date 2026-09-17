#ifndef __VEXT_CORE_CUDA_OPS_ELEMENTWISE_BINARY_CUH__
#define __VEXT_CORE_CUDA_OPS_ELEMENTWISE_BINARY_CUH__

#include <iostream>
#include <vector>

#include <cuda/std/cmath>
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

struct BinaryWithBroadcastMeta
{
	std::uint32_t dims[MAX_RANK];
	std::uint32_t strides[MAX_RANK];
	std::uint32_t size = 0;
};

template <Op Kp, EvaluationMode Mp, typename T1, typename T2, typename T3, typename Dp = core::no_value_t>
requires core::BinaryOperation<Kp>
__global__ void
binary(
	T1*       out,
	const T2* a,
	const T3* __restrict__ b,
	const std::uint32_t N,
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
							out[i] = a[i] + (b[i] + noise(maybe_descriptor, i));
						}
					else
						{
							out[i] = a[i] + b[i];
						}
				}
			else if constexpr(Kp == Op::SUB)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = a[i] - (b[i] + noise(maybe_descriptor, i));
						}
					else
						{
							out[i] = a[i] - b[i];
						}
				}
			else if constexpr(Kp == Op::MUL)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = a[i] * (b[i] + noise(maybe_descriptor, i));
						}
					else
						{
							out[i] = a[i] * b[i];
						}
				}
			else if constexpr(Kp == Op::DIV)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = a[i] / (b[i] + noise(maybe_descriptor, i));
						}
					else
						{
							out[i] = a[i] / b[i];
						}
				}
			else if constexpr(Kp == Op::POW)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = ::cuda::std::pow(a[i], b[i] + noise(maybe_descriptor, i));
						}
					else
						{
							out[i] = ::cuda::std::pow(a[i], b[i]);
						}
				}
			else if constexpr(Kp == Op::MIN)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							const ::cuda::std::common_type_t<T3, float> value = b[i] + noise(maybe_descriptor, i);
							out[i]                                            = (value < a[i]) ? value : a[i];
						}
					else
						{
							out[i] = (b[i] < a[i]) ? b[i] : a[i];
						}
				}
			else if constexpr(Kp == Op::MAX)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							const ::cuda::std::common_type_t<T3, float> value = b[i] + noise(maybe_descriptor, i);
							out[i]                                            = (a[i] < value) ? value : a[i];
						}
					else
						{
							out[i] = (a[i] < b[i]) ? b[i] : a[i];
						}
				}
			else if constexpr(Kp == Op::PRELU)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							const T1 value    = static_cast<T1>(a[i]);
							const T1 positive = (T1{ 0 } < value) ? value : T1{ 0 };
							const T1 negative = (value < T1{ 0 }) ? value : T1{ 0 };
							out[i]            = positive + (b[i] + noise(maybe_descriptor, i)) * negative;
						}
					else
						{
							const T1 value    = static_cast<T1>(a[i]);
							const T1 positive = (T1{ 0 } < value) ? value : T1{ 0 };
							const T1 negative = (value < T1{ 0 }) ? value : T1{ 0 };
							out[i]            = positive + b[i] * negative;
						}
				}
		}
}

template <Op Kp, EvaluationMode Mp, typename T1, typename T2, typename T3, typename Dp = core::no_value_t>
requires core::BinaryOperation<Kp>
__global__ void
binary_with_broadcast(
	T1*       out,
	const T2* a,
	const T3* __restrict__ b,
	const std::uint32_t           N,
	const BinaryWithBroadcastMeta meta,
	const Dp                      maybe_descriptor)
{
	const std::uint32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
	const std::uint32_t stride = blockDim.x * gridDim.x;

	for(std::uint32_t i = tid; i < N; i += stride)
		{
			std::uint32_t tmp      = i;
			std::uint32_t b_offset = 0;

			for(std::uint32_t j = meta.size - 1;; --j)
				{
					const std::uint32_t index_j = tmp % meta.dims[j];

					tmp /= meta.dims[j];
					b_offset += index_j * meta.strides[j];

					if(j == 0)
						{
							break;
						}
				}

			if constexpr(Kp == Op::ADD)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = a[i] + (b[b_offset] + noise(maybe_descriptor, b_offset));
						}
					else
						{
							out[i] = a[i] + b[b_offset];
						}
				}
			else if constexpr(Kp == Op::SUB)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = a[i] - (b[b_offset] + noise(maybe_descriptor, b_offset));
						}
					else
						{
							out[i] = a[i] - b[b_offset];
						}
				}
			else if constexpr(Kp == Op::MUL)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = a[i] * (b[b_offset] + noise(maybe_descriptor, b_offset));
						}
					else
						{
							out[i] = a[i] * b[b_offset];
						}
				}
			else if constexpr(Kp == Op::DIV)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = a[i] / (b[b_offset] + noise(maybe_descriptor, b_offset));
						}
					else
						{
							out[i] = a[i] / b[b_offset];
						}
				}
			else if constexpr(Kp == Op::POW)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = ::cuda::std::pow(a[i], b[b_offset] + noise(maybe_descriptor, b_offset));
						}
					else
						{
							out[i] = ::cuda::std::pow(a[i], b[b_offset]);
						}
				}
			else if constexpr(Kp == Op::MIN)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							const ::cuda::std::common_type_t<T3, float> value = b[b_offset] + noise(maybe_descriptor, b_offset);
							out[i]                                            = (value < a[i]) ? value : a[i];
						}
					else
						{
							out[i] = (b[b_offset] < a[i]) ? b[b_offset] : a[i];
						}
				}
			else if constexpr(Kp == Op::MAX)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							const ::cuda::std::common_type_t<T3, float> value = b[b_offset] + noise(maybe_descriptor, b_offset);
							out[i]                                            = (a[i] < value) ? value : a[i];
						}
					else
						{
							out[i] = (a[i] < b[b_offset]) ? b[b_offset] : a[i];
						}
				}
			else if constexpr(Kp == Op::PRELU)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							const T1 value    = static_cast<T1>(a[i]);
							const T1 positive = (T1{ 0 } < value) ? value : T1{ 0 };
							const T1 negative = (value < T1{ 0 }) ? value : T1{ 0 };
							out[i]            = positive + (b[b_offset] + noise(maybe_descriptor, b_offset)) * negative;
						}
					else
						{
							const T1 value    = static_cast<T1>(a[i]);
							const T1 positive = (T1{ 0 } < value) ? value : T1{ 0 };
							const T1 negative = (value < T1{ 0 }) ? value : T1{ 0 };
							out[i]            = positive + b[b_offset] * negative;
						}
				}
		}
}

}

namespace vext::core::cuda::ops
{

template <Op Kp, EvaluationMode Mp = EvaluationMode::PLAIN, typename T1, typename T2, typename T3>
requires core::BinaryOperation<Kp>
void
binary(
	T1*                 out,
	const T2*           a,
	const T3*           b,
	const std::uint32_t N)
{
	constexpr std::uint32_t block_size = 256;
	const std::uint32_t     grid_size  = (N + block_size - 1) / block_size;

	if constexpr(Mp == EvaluationMode::PERTURBED)
		{
			kernel::binary<Kp, Mp><<<grid_size, block_size>>>(out, a, b, N, sequentional_noise_descriptor());
		}
	else
		{
			kernel::binary<Kp, Mp><<<grid_size, block_size>>>(out, a, b, N, core::no_value);
		}

	CUDA_CHECK(cudaGetLastError());
}

template <Op Kp, EvaluationMode Mp = EvaluationMode::PLAIN, typename T1, typename T2, typename T3>
requires core::BinaryOperation<Kp>
void
binary_with_broadcast(
	T1*                               out,
	const T2*                         a,
	const T3*                         b,
	const std::uint32_t               N,
	const std::vector<std::uint32_t>& dims,
	const std::vector<std::uint32_t>& strides)
{
	constexpr std::uint32_t block_size = 256;
	const std::uint32_t     grid_size  = (N + block_size - 1) / block_size;

	kernel::BinaryWithBroadcastMeta meta = { .size = static_cast<std::uint32_t>(dims.size()) };

	for(std::uint32_t i = 0; i < meta.size; ++i)
		{
			meta.dims[i]    = dims[i];
			meta.strides[i] = strides[i];
		}

	if constexpr(Mp == EvaluationMode::PERTURBED)
		{
			const NoiseDescriptor& descriptor = sequentional_noise_descriptor();
			kernel::binary_with_broadcast<Kp, Mp><<<grid_size, block_size>>>(out, a, b, N, meta, descriptor);
		}
	else
		{
			kernel::binary_with_broadcast<Kp, Mp><<<grid_size, block_size>>>(out, a, b, N, meta, core::no_value);
		}

	CUDA_CHECK(cudaGetLastError());
}

}

#undef CUDA_CHECK

#endif
