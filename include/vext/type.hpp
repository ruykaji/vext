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

enum class Reduce : std::uint8_t
{
	ADD = 0,
	MUL,
	MIN,
	MAX,
	MEAN
};

enum class Mutation : std::uint8_t
{
	IN_PLACE = 0,
	COPY
};

}

#endif
