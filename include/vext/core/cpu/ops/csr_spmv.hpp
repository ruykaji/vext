#ifndef __VEXT_CORE_CPU_OPS_CSR_SPMV_HPP__
#define __VEXT_CORE_CPU_OPS_CSR_SPMV_HPP__

#include <algorithm>
#include <cmath>

#include <vext/core/type.hpp>
#include <vext/type.hpp>

namespace vext::core::cpu::ops
{

template <CSRSpMVOp Kp, typename T1, typename T2, typename T3>
void
csr_spmv(
	T1* __restrict__ y,
	const T2* __restrict__ A,
	const std::uint32_t* __restrict__ head,
	const std::uint32_t* __restrict__ tail,
	const T3* __restrict__ x,
	const std::uint32_t N)
{
	for(std::uint32_t i = 0; i < N; ++i)
		{
			const std::uint32_t start = head[i];
			const std::uint32_t end   = head[i + 1];

			T1 accumulator = 0;

			if constexpr(Kp == CSRSpMVOp::PROD)
				{
					accumulator = 1;
				}
			else if constexpr(Kp == CSRSpMVOp::MIN)
				{
					accumulator = std::numeric_limits<T1>::max();
				}
			else if constexpr(Kp == CSRSpMVOp::MAX)
				{
					accumulator = std::numeric_limits<T1>::lowest();
				}

			for(std::uint32_t h = start; h < end; ++h)
				{
					const T1 prod = A[h] * x[tail[h]];

					if constexpr(Kp == CSRSpMVOp::PROD)
						{
							accumulator *= prod;
						}
					else if constexpr(Kp == CSRSpMVOp::MIN)
						{
							accumulator = std::min<T1>(accumulator, prod);
						}
					else if constexpr(Kp == CSRSpMVOp::MAX)
						{
							accumulator = std::max<T1>(accumulator, prod);
						}
					else
						{
							accumulator += prod;
						}
				}

			if constexpr(Kp == CSRSpMVOp::MEAN)
				{
					const float scale = 1.0f / static_cast<float>(end - start);
					y[i]              = accumulator * scale;
				}
			else if constexpr(Kp == CSRSpMVOp::VAR || Kp == CSRSpMVOp::STD)
				{
					const float scale = 1.0f / static_cast<float>(end - start);
					const float mean  = accumulator * scale;

					float dispertion = 0.0f;

					for(std::uint32_t h = start; h < end; ++h)
						{
							const T1    prod = A[h] * x[tail[h]];
							const float diff = prod - mean;

							dispertion += diff * diff;
						}

					if constexpr(Kp == CSRSpMVOp::VAR)
						{
							y[i] = dispertion * scale;
						}
					else
						{
							y[i] = std::sqrt(dispertion * scale);
						}
				}
			else
				{
					y[i] = accumulator;
				}
		}
}

}

#endif
