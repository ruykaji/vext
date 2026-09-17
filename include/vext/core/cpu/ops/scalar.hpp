#ifndef __VEXT_CORE_CPU_SCALAR_HPP__
#define __VEXT_CORE_CPU_SCALAR_HPP__

#include <vext/core/cpu/noise.hpp>
#include <vext/core/type.hpp>
#include <vext/type.hpp>

namespace vext::core::cpu::ops
{

template <Op Kp, ParameterMode Mp, typename T1, typename T2, typename T3>
void
scalar(
	T1*                 out,
	const T2*           src,
	const std::uint32_t N,
	const T3            value)
{
	for(std::uint32_t i = 0; i < N; ++i)
		{
			if constexpr(Kp == Op::ADD)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = (src[i] + noise(i)) + value;
						}
					else
						{
							out[i] = src[i] + value;
						}
				}
			else if constexpr(Kp == Op::SUB)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = (src[i] + noise(i)) - value;
						}
					else
						{
							out[i] = src[i] - value;
						}
				}
			if constexpr(Kp == Op::MUL)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = (src[i] + noise(i)) * value;
						}
					else
						{
							out[i] = src[i] * value;
						}
				}
			else if constexpr(Kp == Op::DIV)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = (src[i] + noise(i)) / value;
						}
					else
						{
							out[i] = src[i] / value;
						}
				}
			else if constexpr(Kp == Op::POW)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = (src[i] + noise(i)) ^ value;
						}
					else
						{
							out[i] = src[i] ^ value;
						}
				}
		}
}

}

#endif
