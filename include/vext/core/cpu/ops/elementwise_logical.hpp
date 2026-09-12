#ifndef __VEXT_CORE_CPU_OPS_ELEMENTWISE_LOGIC_HPP__
#define __VEXT_CORE_CPU_OPS_ELEMENTWISE_LOGIC_HPP__

#include <cstring>

#include <vext/core/type.hpp>
#include <vext/type.hpp>

namespace vext::core::cpu::ops
{

template <Op Kp, typename T1, typename T2>
requires core::LogicalOperation<Kp>
static void
logical(
	std::uint8_t* __restrict__ out,
	const T1* __restrict__ a,
	const T2* __restrict__ b,
	const std::uint32_t N)
{
	if(a == b)
		{
			if constexpr(Kp == Op::LESS || Kp == Op::GREATER)
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
					if constexpr(Kp == Op::EQUAL)
						{
							out[i] = a[i] == b[i];
						}
					else if constexpr(Kp == Op::NOT_EQUAL)
						{
							out[i] = a[i] != b[i];
						}
					else if constexpr(Kp == Op::LESS)
						{
							out[i] = a[i] < b[i];
						}
					else if constexpr(Kp == Op::LESS_EQUAL)
						{
							out[i] = a[i] <= b[i];
						}
					else if constexpr(Kp == Op::GREATER)
						{
							out[i] = a[i] > b[i];
						}
					else if constexpr(Kp == Op::GREATER_EQUAL)
						{
							out[i] = a[i] >= b[i];
						}
				}
		}
}

}

#endif
