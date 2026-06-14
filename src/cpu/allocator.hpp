#ifndef __VEXT_ALLOCATOR_CPU_HPP__
#define __VEXT_ALLOCATOR_CPU_HPP__

#include <cstdint>

namespace vext::cpu
{

class Allocator
{
public:
	static void*
	allocate(
		const std::uint64_t requested_size);

	static void
	deallocate(
		void* ptr);

	static void
	free();
};

}

#endif
