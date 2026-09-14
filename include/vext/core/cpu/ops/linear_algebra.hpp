#ifndef __VEXT_CORE_CPU_OPS_LINEAR_ALGEBRA_HPP__
#define __VEXT_CORE_CPU_OPS_LINEAR_ALGEBRA_HPP__

#include <vext/core/cpu/noise.hpp>
#include <vext/core/type.hpp>
#include <vext/type.hpp>

namespace vext::core::cpu::ops
{

template <ParameterMode Mp, typename T1, typename T2, typename T3>
void
matmul(
	T1* __restrict__ out,
	const T2* __restrict__ a,
	const T3* __restrict__ b,
	const std::uint32_t M,
	const std::uint32_t P,
	const std::uint32_t N)
{
	for(std::uint32_t m = 0; m < M; ++m)
		{
			for(std::uint32_t p = 0; p < P; ++p)
				{
					const T1 a_value = a[m * P + p];

					for(std::uint32_t n = 0; n < N; ++n)
						{
							if constexpr(Mp == ParameterMode::PERTURBED)
								{
									out[m * N + n] += a_value * (b[p * N + n] + noise(p * N + n));
								}
							else
								{
									out[m * N + n] += a_value * b[p * N + n];
								}
						}
				}
		}
}

}

#endif
