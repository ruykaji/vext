#ifndef __VEXT_TYPE_HPP__
#define __VEXT_TYPE_HPP__

#include <cstdint>
#include <vector>

namespace vext
{

enum class Backend : std::uint8_t
{
	CPU = 0,
	CUDA
};

enum class Op : std::uint8_t
{
	ABS = 0,
	SIN,
	COS,
	TANH,
	NEG,
	EXP,
	LOG,
	SQRT,
	SQUARE,
	ROUND,
	SIGMOID,
	SOFT_RELU,
	RELU,
	SOFTMAX,
	SOFTMIN,
	LOGSOFTMAX,
	LEAKY_RELU,
	ELU,
	SWISH,
	LINEAR,
	CLIP,
	POW,
	ADD,
	SUB,
	MUL,
	DIV,
	MIN,
	MAX,
	PRELU,
	EQUAL,
	NOT_EQUAL,
	LESS,
	LESS_EQUAL,
	GREATER,
	GREATER_EQUAL,
	SUM,
	MEAN,
	PROD,
	STD,
	VAR,
	L2_NORM
};

using Axes = std::vector<std::int32_t>;

enum class Mutation : std::uint8_t
{
	IN_PLACE = 0,
	COPY
};

enum class Noise : std::uint8_t
{
	NONE = 0,
	SEED_HASH
};

}

#endif
