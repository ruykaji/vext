#ifndef __VEXT_CORE_CUDA_OPERATIONS_MEMORY_CUH__
#define __VEXT_CORE_CUDA_OPERATIONS_MEMORY_CUH__

#include <cstdint>
#include <iostream>

#include <cuda_runtime.h>

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

namespace vext::core::cuda::operations
{

template <typename Tp>
void
memset(
	Tp*                 ptr,
	const Tp            value,
	const std::uint32_t size)
{
	CUDA_CHECK(cudaMemset(ptr, value, size * sizeof(Tp)));
}

template <typename Tp, Backend B1, Backend B2>
void
memcpy(
	Tp*                 dst,
	const Tp*           src,
	const std::uint32_t count)
{
	if(B1 == Backend::CUDA)
		{
			if constexpr(B2 == Backend::CPU)
				{
					CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(Tp), cudaMemcpyKind::cudaMemcpyHostToDevice));
				}
			else
				{
					CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(Tp), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
				}
		}
	else if constexpr(B2 == Backend::CUDA)
		{
			CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(Tp), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		}
}

template <typename Tp>
Tp
memget(
	Tp*                 ptr,
	const std::uint32_t index)
{
	Tp value;
	CUDA_CHECK(cudaMemcpy(&value, ptr + index, sizeof(Tp), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	return value;
}

}

#undef CUDA_CHECK

#endif
