#ifndef __VEXT_CORE_TYPE_HPP__
#define __VEXT_CORE_TYPE_HPP__

#include <cstdint>

namespace vext::core
{

enum class UnaryOperation : std::uint8_t
{
	NONE = 0,
	ABS,
	SIN,
	COS,
	NEG,
	EXP,
	LOG,
	SQRT,
	SIGMOID
};

enum class BinaryOperation : std::uint8_t
{
	NONE = 0,
	ADD,
	SUB,
	MUL,
	DIV,
	POW,
	MIN,
	MAX
};

enum class LogicOperation : std::uint8_t
{
	NONE = 0,
	EQUAL,
	NOT_EQUAL,
	LESS,
	LESS_EQUAL,
	GREATER,
	GREATER_EQUAL
};

enum class ReductionOperation : std::uint8_t
{
	NONE = 0,
	SUM,
	MEAN,
	MAX,
	MIN,
	PROD,
	STD,
	VAR
};

}

#endif
