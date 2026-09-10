#ifndef __VEXT_CORE_CPU_OPS_ELEMENTWISE_LOGIC_HPP__
#define __VEXT_CORE_CPU_OPS_ELEMENTWISE_LOGIC_HPP__

#include <cstring>

#include <vext/core/type.hpp>
#include <vext/type.hpp>

namespace vext::core::cpu::ops
{

template <LogicOp Kp, typename T1, typename T2>
static void
logical(
	std::uint8_t* __restrict__ out,
	const T1* __restrict__ a,
	const T2* __restrict__ b,
	const std::uint32_t N)
{
	if(a == b)
		{
			if constexpr(Kp == LogicOp::LESS || Kp == LogicOp::GREATER)
				{
					std::memset(out, 0, N * sizeof(std::uint8_t));
				}
			else
				{
					std::memset(out, 1, N * sizeof(std::uint8_t));
				}
		}
	else
		{
			for(std::uint32_t i = 0; i < N; ++i)
				{
					if constexpr(Kp == LogicOp::EQUAL)
						{
							out[i] = a[i] == b[i];
						}
					else if constexpr(Kp == LogicOp::NOT_EQUAL)
						{
							out[i] = a[i] != b[i];
						}
					else if constexpr(Kp == LogicOp::LESS)
						{
							out[i] = a[i] < b[i];
						}
					else if constexpr(Kp == LogicOp::LESS_EQUAL)
						{
							out[i] = a[i] <= b[i];
						}
					else if constexpr(Kp == LogicOp::GREATER)
						{
							out[i] = a[i] > b[i];
						}
					else if constexpr(Kp == LogicOp::GREATER_EQUAL)
						{
							out[i] = a[i] >= b[i];
						}
				}
		}
}

}

#endif
