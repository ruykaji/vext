#ifndef __VEXT_TYPE_HPP__
#define __VEXT_TYPE_HPP__

#include <cstdint>

namespace vext
{

enum class Backend : std::uint8_t
{
	CPU = 0,
	CUDA
};

enum class ReduceKind : std::uint8_t
{
	NONE = 0,
	ADD,
	MUL,
	MIN,
	MAX,
	MEAN
};

enum class InitializationKind : std::uint8_t
{
	NONE = 0,
	XAVIER_NORMAL,
	XAVIER_UNIFORM,
	KAIMING_NORMAL,
	KAIMING_UNIFORM,
};

}

#endif
