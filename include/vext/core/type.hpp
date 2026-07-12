#ifndef __VEXT_CORE_TYPE_HPP__
#define __VEXT_CORE_TYPE_HPP__

#include <cstdint>
#include <limits>

namespace vext::core
{

inline constexpr std::uint32_t MIN_RANK   = 1;
inline constexpr std::uint32_t MAX_RANK   = 32;
inline constexpr std::uint32_t MIN_LENGTH = 1;
inline constexpr std::uint32_t MAX_LENGTH = std::numeric_limits<std::uint32_t>::max() / MAX_RANK;

template <auto...>
inline constexpr bool dependent_false = false;

enum class UnaryOperation : std::uint8_t
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

enum class BinaryOperation : std::uint8_t
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

enum class LogicOperation : std::uint8_t
{
	EQUAL = 0,
	NOT_EQUAL,
	LESS,
	LESS_EQUAL,
	GREATER,
	GREATER_EQUAL
};

enum class ReductionOperation : std::uint8_t
{
	SUM = 0,
	MEAN,
	MAX,
	MIN,
	PROD,
	STD,
	VAR
};

enum class CSRScatterOperation : std::uint8_t
{
	SUM = 0,
	MEAN,
	MAX,
	MIN,
	PROD,
	STD,
	VAR
};

}

#endif
