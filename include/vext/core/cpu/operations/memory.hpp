#ifndef __VEXT_CORE_CUDA_OPERATIONS_MEMORY_HPP__
#define __VEXT_CORE_CUDA_OPERATIONS_MEMORY_HPP__

#include <cstdint>
#include <cstring>

#include <vext/type.hpp>

namespace vext::core::cpu::operations
{

template <typename Tp>
void
memset(
	Tp*                 ptr,
	const Tp            value,
	const std::uint32_t size)
{
	std::memset(ptr, value, size * sizeof(Tp));
}

template <typename Tp>
void
memcpy(
	Tp*                 dst,
	const Tp*           src,
	const std::uint32_t count)
{
	std::memcpy(dst, src, count * sizeof(Tp));
}

}

#endif
