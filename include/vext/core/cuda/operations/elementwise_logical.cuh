#ifndef __VEXT_CORE_CUDA_OPERATIONS_ELEMENTWISE_LOGIC_CUH__
#define __VEXT_CORE_CUDA_OPERATIONS_ELEMENTWISE_LOGIC_CUH__

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

template <LogicOperation Kp, typename T1, typename T2>
__global__ void
logical(
	std::uint8_t* __restrict__ out,
	const T1* __restrict__ a,
	const T2* __restrict__ b,
	const std::uint32_t N)
{
	const std::uint32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
	const std::uint32_t stride = blockDim.x * gridDim.x;

	for(std::uint32_t i = tid; i < N; i += stride)
		{
			if constexpr(Kp == LogicOperation::EQUAL)
				{
					out[i] = a[i] == b[i];
				}
			else if constexpr(Kp == LogicOperation::NOT_EQUAL)
				{
					out[i] = a[i] != b[i];
				}
			else if constexpr(Kp == LogicOperation::LESS)
				{
					out[i] = a[i] < b[i];
				}
			else if constexpr(Kp == LogicOperation::LESS_EQUAL)
				{
					out[i] = a[i] <= b[i];
				}
			else if constexpr(Kp == LogicOperation::GREATER)
				{
					out[i] = a[i] > b[i];
				}
			else if constexpr(Kp == LogicOperation::GREATER_EQUAL)
				{
					out[i] = a[i] >= b[i];
				}
		}
}

}

namespace vext::core::cuda::operations
{

template <LogicOperation Kp, typename T1, typename T2>
void
logical(
	std::uint8_t*       out,
	const T1*           a,
	const T2*           b,
	const std::uint32_t N)
{
	constexpr std::uint32_t block_size = 256;
	const std::uint32_t     grid_size  = (N + block_size - 1) / block_size;

	if(a == b)
		{
			if constexpr(Kp == LogicOperation::LESS || Kp == LogicOperation::GREATER)
				{
					CUDA_CHECK(cudaMemset(out, 0, N * sizeof(std::uint8_t)));
				}
			else
				{
					CUDA_CHECK(cudaMemset(out, 1, N * sizeof(std::uint8_t)));
				}
		}
	else
		{
			kernel::logical<Kp, T1, T2><<<grid_size, block_size>>>(out, a, b, N);
			CUDA_CHECK(cudaGetLastError());
		}
}

}

#undef CUDA_CHECK

#endif
