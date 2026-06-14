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

enum class ParameterInit : std::uint8_t
{
	XAVIER_NORMAL = 0,
	XAVIER_UNIFORM,
	KAIMING_NORMAL,
	KAIMING_UNIFORM,
};

}

#endif
