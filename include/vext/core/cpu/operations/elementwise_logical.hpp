#ifndef __VEXT_CORE_CPU_OPERATIONS_ELEMENTWISE_LOGIC_HPP__
#define __VEXT_CORE_CPU_OPERATIONS_ELEMENTWISE_LOGIC_HPP__

#include <vext/core/type.hpp>

namespace vext::core::cpu::operations
{

template <LogicOperation Kp, typename T1, typename T2>
static void
logical(
	std::uint8_t*       out,
	const T1*           a,
	const T2*           b,
	const std::uint64_t N)
{
	for(std::uint64_t i = 0; i < N; ++i)
		{
			if constexpr(Kp == LogicOperation::EQUAL)
				{
					out[i] = a[i] == b[i];
				}
			else if constexpr(Kp == LogicOperation::NOT_EQUAL)
				{
					out[i] = a[i] != b[i];
				}
			else if constexpr(Kp == LogicOperation::LESS)
				{
					out[i] = a[i] < b[i];
				}
			else if constexpr(Kp == LogicOperation::LESS_EQUAL)
				{
					out[i] = a[i] <= b[i];
				}
			else if constexpr(Kp == LogicOperation::GREATER)
				{
					out[i] = a[i] > b[i];
				}
			else if constexpr(Kp == LogicOperation::GREATER_EQUAL)
				{
					out[i] = a[i] >= b[i];
				}
		}
}

}

#endif
