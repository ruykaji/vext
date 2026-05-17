#ifndef __VEXT_TYPE_HPP__
#define __VEXT_TYPE_HPP__

#include <cstdint>

namespace vext::type
{

enum class Backend : std::uint8_t
{
	CPU = 0,
	CUDA
};

}

#endif
