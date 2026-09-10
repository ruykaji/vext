#ifndef __VEXT_CORE_CUDA_OPS_ELEMENTWISE_BINARY_CUH__
#define __VEXT_CORE_CUDA_OPS_ELEMENTWISE_BINARY_CUH__

#include <iostream>
#include <vector>

#include <cuda/std/algorithm>
#include <cuda/std/cmath>
#include <cuda_runtime.h>

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

template <BinaryOp Kp, typename T1, typename T2, typename T3>
__global__ void
binary(
	T1*       out,
	const T2* a,
	const T3* __restrict__ b,
	const std::uint32_t N)
{
	const std::uint32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
	const std::uint32_t stride = blockDim.x * gridDim.x;

	for(std::uint32_t i = tid; i < N; i += stride)
		{
			if constexpr(Kp == BinaryOp::ADD)
				{
					out[i] = a[i] + b[i];
				}
			else if constexpr(Kp == BinaryOp::SUB)
				{
					out[i] = a[i] - b[i];
				}
			else if constexpr(Kp == BinaryOp::MUL)
				{
					out[i] = a[i] * b[i];
				}
			else if constexpr(Kp == BinaryOp::DIV)
				{
					out[i] = a[i] / b[i];
				}
			else if constexpr(Kp == BinaryOp::POW)
				{
					out[i] = ::cuda::std::pow(a[i], b[i]);
				}
			else if constexpr(Kp == BinaryOp::MIN)
				{
					out[i] = ::cuda::std::min(a[i], b[i]);
				}
			else if constexpr(Kp == BinaryOp::MAX)
				{
					out[i] = ::cuda::std::max(a[i], b[i]);
				}
			else if constexpr(Kp == BinaryOp::PRELU)
				{
					out[i] = ::cuda::std::max<T1>(0, a[i]) + b[i] * ::cuda::std::min<T1>(0, a[i]);
				}
		}
}

template <BinaryOp Kp, typename T1, typename T2, typename T3>
__global__ void
binary_with_broadcast(
	T1*       out,
	const T2* a,
	const T3* __restrict__ b,
	const std::uint32_t           N,
	const BinaryWithBroadcastMeta meta)
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

			if constexpr(Kp == BinaryOp::ADD)
				{

					out[i] = a[i] + b[b_offset];
				}
			else if constexpr(Kp == BinaryOp::SUB)
				{

					out[i] = a[i] - b[b_offset];
				}
			else if constexpr(Kp == BinaryOp::MUL)
				{

					out[i] = a[i] * b[b_offset];
				}
			else if constexpr(Kp == BinaryOp::DIV)
				{

					out[i] = a[i] / b[b_offset];
				}
			else if constexpr(Kp == BinaryOp::POW)
				{

					out[i] = ::cuda::std::pow(a[i], b[b_offset]);
				}
			else if constexpr(Kp == BinaryOp::MIN)
				{

					out[i] = ::cuda::std::min(a[i], b[b_offset]);
				}
			else if constexpr(Kp == BinaryOp::MAX)
				{

					out[i] = ::cuda::std::max(a[i], b[b_offset]);
				}
			else if constexpr(Kp == BinaryOp::PRELU)
				{
					out[i] = ::cuda::std::max<T1>(0, a[i]) + b[b_offset] * ::cuda::std::min<T1>(0, a[i]);
				}
		}
}

}

namespace vext::core::cuda::ops
{

template <BinaryOp Kp, typename T1, typename T2, typename T3>
void
binary(
	T1*                 out,
	const T2*           a,
	const T3*           b,
	const std::uint32_t N)
{
	constexpr std::uint32_t block_size = 256;
	const std::uint32_t     grid_size  = (N + block_size - 1) / block_size;

	kernel::binary<Kp><<<grid_size, block_size>>>(out, a, b, N);
	CUDA_CHECK(cudaGetLastError());
}

template <BinaryOp Kp, typename T1, typename T2, typename T3>
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

	kernel::binary_with_broadcast<Kp><<<grid_size, block_size>>>(out, a, b, N, meta);
	CUDA_CHECK(cudaGetLastError());
}

}

#undef CUDA_CHECK

#endif
