#ifndef __VEXT_CORE_CPU_OPERATIONS_LINEAR_ALGEBRA_HPP__
#define __VEXT_CORE_CPU_OPERATIONS_LINEAR_ALGEBRA_HPP__

#include <cmath>
#include <vector>

#include <vext/core/type.hpp>

namespace vext::core::cpu::operations
{

template <typename T1, typename T2>
void
matmul(
	std::common_type_t<T1, T2>* out,
	const T1*                   a,
	const T2*                   b,
	const std::uint32_t         M,
	const std::uint32_t         P,
	const std::uint32_t         N)
{
	for(std::uint32_t m = 0; m < M; ++m)
		{
			for(std::uint32_t p = 0; p < P; ++p)
				{
					const T1 a_value = a[m * P + p];

					for(std::uint32_t n = 0; n < N; ++n)
						{
							out[m * N + n] += a_value * b[p * N + n];
						}
				}
		}
}

}

#endif
