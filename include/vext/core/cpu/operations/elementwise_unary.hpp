#ifndef __VEXT_CORE_CPU_OPERATIONS_ELEMENTWISE_UNARY_HPP__
#define __VEXT_CORE_CPU_OPERATIONS_ELEMENTWISE_UNARY_HPP__

#include <cmath>

#include <vext/core/type.hpp>

namespace vext::core::cpu::operations
{

template <UnaryOperation Kp, typename T1>
static void
unary(
	T1*                 out,
	const std::uint64_t N)
{
	for(std::uint64_t i = 0; i < N; ++i)
		{
			if constexpr(Kp == UnaryOperation::ABS)
				{
					out[i] = std::abs(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::SIN)
				{
					out[i] = std::sin(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::COS)
				{
					out[i] = std::cos(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::NEG)
				{
					out[i] = -out[i];
				}
			else if constexpr(Kp == UnaryOperation::EXP)
				{
					out[i] = std::exp(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::LOG)
				{
					out[i] = std::log(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::SQRT)
				{
					out[i] = std::sqrt(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::SIGMOID)
				{
					out[i] = 1.0f / (1.0f + std::exp(-out[i]));
				}
		}
}

}

#endif
