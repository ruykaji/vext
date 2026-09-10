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

enum class UnaryOp : std::uint8_t
{
	/** -- NO PARAMETERS REQUIRED -- */
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
	/** -- REQUIRES ALPHA -- */
	LEAKY_RELU,
	ELU,
	SWISH,
	/** -- REQUIRES ALPHA AND BETA -- */
	LINEAR,
	CLIP,
	POW
};

enum class BinaryOp : std::uint8_t
{
	ADD = 0,
	SUB,
	MUL,
	DIV,
	POW,
	MIN,
	MAX,
	PRELU
};

enum class LogicOp : std::uint8_t
{
	EQUAL = 0,
	NOT_EQUAL,
	LESS,
	LESS_EQUAL,
	GREATER,
	GREATER_EQUAL
};

enum class ReductionOp : std::uint8_t
{
	SUM = 0,
	MEAN,
	MAX,
	MIN,
	PROD,
	STD,
	VAR,
	L2_NORM
};

enum class CSRScatterOp : std::uint8_t
{
	SUM = 0,
	MEAN,
	MAX,
	MIN,
	PROD,
	STD,
	VAR
};

enum class CSRSpMVOp : std::uint8_t
{
	SUM = 0,
	MEAN,
	MAX,
	MIN,
	PROD,
	STD,
	VAR
};

enum class Mutation : std::uint8_t
{
	IN_PLACE = 0,
	COPY
};

}

#endif
