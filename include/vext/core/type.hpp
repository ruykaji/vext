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

template <typename Tp>
concept Arithmetic = std::is_arithmetic_v<Tp>;

/** === Reduction return type deduction == */

template <ReductionOp Kp>
inline constexpr bool is_float_reducing = Kp == ReductionOp::MEAN || Kp == ReductionOp::VAR || Kp == ReductionOp::STD || Kp == ReductionOp::L2_NORM;

template <ReductionOp Kp, typename Tp>
using ReductionOut = std::conditional_t<is_float_reducing<Kp>, float, Tp>;

/** === CSR Scatter return type deduction == */

template <CSRScatterOp Kp>
inline constexpr bool is_float_csr_scatter = Kp == CSRScatterOp::MEAN || Kp == CSRScatterOp::VAR || Kp == CSRScatterOp::STD;

template <CSRScatterOp Kp, typename Tp>
using CSRScatterOut = std::conditional_t<is_float_csr_scatter<Kp>, float, Tp>;

/** === CSR SpMV return type deduction == */

template <CSRSpMVOp Kp>
inline constexpr bool is_float_csr_spmv = Kp == CSRSpMVOp::MEAN || Kp == CSRSpMVOp::VAR || Kp == CSRSpMVOp::STD;

template <CSRSpMVOp Kp, typename Tp>
using CSRSpMVOut = std::conditional_t<is_float_csr_spmv<Kp>, float, Tp>;

}

#endif
