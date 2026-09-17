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

enum class Mutation : std::uint8_t
{
	OUT_OF_PLACE = 0,
	IN_PLACE
};

enum class EvaluationMode : std::uint8_t
{
	PLAIN = 0,
	PERTURBED,
	DIFFERENTIABLE
};

}

#endif
