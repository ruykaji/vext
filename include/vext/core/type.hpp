#ifndef __VEXT_CORE_TYPE_HPP__
#define __VEXT_CORE_TYPE_HPP__

#include <cstdint>

namespace vext::core
{

enum class UnaryOperationKind : std::uint8_t
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

enum class BinaryOperationKind : std::uint8_t
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

enum class LogicOperationKind : std::uint8_t
{
	NONE = 0,
	EQUAL,
	NOT_EQUAL,
	LESS,
	LESS_EQUAL,
	GREATER,
	GREATER_EQUAL
};

enum class ReductionOperationKind : std::uint8_t
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
