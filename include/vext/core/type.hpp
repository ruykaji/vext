#ifndef __VEXT_CORE_TYPE_HPP__
#define __VEXT_CORE_TYPE_HPP__

#include <cstdint>
#include <limits>
#include <type_traits>

#include <vext/type.hpp>

namespace vext::core
{

inline constexpr std::uint32_t MIN_RANK   = 1;
inline constexpr std::uint32_t MAX_RANK   = 32;
inline constexpr std::uint32_t MIN_LENGTH = 1;
inline constexpr std::uint32_t MAX_LENGTH = std::numeric_limits<std::uint32_t>::max() / MAX_RANK;

template <auto...>
inline constexpr bool dependent_false = false;

/** === Concept for checking argument types === */
template <typename Tp>
concept Arithmetic = std::is_arithmetic_v<Tp>;

/** === Template default for case when no value was provided === */

struct no_value_t
{
};

inline constexpr no_value_t no_value{};

/** === Checks if type is Tensor instantiation === */

template <typename, template <typename, Backend> class Template>
struct is_tensor_instantiation : std::false_type
{
};

template <typename Tp, Backend Bp, template <typename, Backend> class Template>
struct is_tensor_instantiation<Template<Tp, Bp>, Template> : std::true_type
{
};

template <typename Tp, Backend Bp, template <typename, Backend> class Template>
inline constexpr bool is_tensor_instatiation_v = is_tensor_instantiation<Template<Tp, Bp>, Template>::value;

/** === Operation category constraints === */

template <Op Kp>
concept UnaryOperation = Kp == Op::ABS || Kp == Op::SIN || Kp == Op::COS || Kp == Op::TANH || Kp == Op::NEG || Kp == Op::EXP || Kp == Op::LOG || Kp == Op::SQRT || Kp == Op::SQUARE || Kp == Op::ROUND || Kp == Op::SIGMOID || Kp == Op::SOFT_RELU || Kp == Op::RELU || Kp == Op::SOFTMAX || Kp == Op::SOFTMIN || Kp == Op::LOGSOFTMAX || Kp == Op::LEAKY_RELU || Kp == Op::ELU || Kp == Op::SWISH || Kp == Op::LINEAR || Kp == Op::CLIP || Kp == Op::POW;

template <Op Kp>
concept BinaryOperation = Kp == Op::ADD || Kp == Op::SUB || Kp == Op::MUL || Kp == Op::DIV || Kp == Op::POW || Kp == Op::MIN || Kp == Op::MAX || Kp == Op::PRELU;

template <Op Kp>
concept LogicalOperation = Kp == Op::EQUAL || Kp == Op::NOT_EQUAL || Kp == Op::LESS || Kp == Op::LESS_EQUAL || Kp == Op::GREATER || Kp == Op::GREATER_EQUAL;

template <Op Kp>
concept ReductionOperation = Kp == Op::SUM || Kp == Op::MEAN || Kp == Op::MAX || Kp == Op::MIN || Kp == Op::PROD || Kp == Op::STD || Kp == Op::VAR || Kp == Op::L2_NORM;

template <Op Kp>
concept SparseReductionOperation = Kp == Op::SUM || Kp == Op::MEAN || Kp == Op::MAX || Kp == Op::MIN || Kp == Op::PROD || Kp == Op::STD || Kp == Op::VAR;

/** === Reduction return type deduction == */

template <Op Kp>
inline constexpr bool is_float_reducing = Kp == Op::MEAN || Kp == Op::VAR || Kp == Op::STD || Kp == Op::L2_NORM;

template <Op Kp, typename Tp>
using ReductionOut = std::conditional_t<is_float_reducing<Kp>, float, Tp>;

/** === CSR Scatter return type deduction == */

template <Op Kp>
inline constexpr bool is_float_csr_scatter = Kp == Op::MEAN || Kp == Op::VAR || Kp == Op::STD;

template <Op Kp, typename Tp>
using CSRScatterOut = std::conditional_t<is_float_csr_scatter<Kp>, float, Tp>;

/** === CSR SpMV return type deduction == */

template <Op Kp>
inline constexpr bool is_float_csr_spmv = Kp == Op::MEAN || Kp == Op::VAR || Kp == Op::STD;

template <Op Kp, typename Tp>
using CSRSpMVOut = std::conditional_t<is_float_csr_spmv<Kp>, float, Tp>;

}

#endif
