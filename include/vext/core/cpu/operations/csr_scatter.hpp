#ifndef __VEXT_CORE_CPU_OPERATIONS_CSR_SCATTER_HPP__
#define __VEXT_CORE_CPU_OPERATIONS_CSR_SCATTER_HPP__

#include <algorithm>
#include <cmath>
#include <vector>

#include <vext/core/type.hpp>

namespace vext::core::cpu::operations
{

template <CSRScatterOperation Kp, typename T1, typename T2>
void
csr_scatter(
	T1* __restrict__ out,
	const T2* __restrict__ src,
	const std::uint32_t* __restrict__ head,
	const std::uint32_t* __restrict__ tail,
	const std::uint32_t N,
	const std::uint32_t S)
{
	std::vector<float> mean_buffer;

	if constexpr(Kp == CSRScatterOperation::VAR || Kp == CSRScatterOperation::STD)
		{
			mean_buffer.resize(S, 0);
		}

	for(std::uint32_t i = 0; i < N; ++i)
		{
			const std::uint32_t start = head[i];
			const std::uint32_t end   = head[i + 1];

			if(start == end)
				{
					continue;
				}

			for(std::uint32_t k = 0; k < S; ++k)
				{
					if constexpr(Kp == CSRScatterOperation::PROD)
						{
							out[i * S + k] = 1;
						}
					else if constexpr(Kp == CSRScatterOperation::MIN)
						{
							out[i * S + k] = std::numeric_limits<T1>::max();
						}
					else if constexpr(Kp == CSRScatterOperation::MAX)
						{
							out[i * S + k] = std::numeric_limits<T1>::lowest();
						}
				}

			for(std::uint32_t h = start; h < end; ++h)
				{
					const std::uint32_t t = tail[h];

					for(std::uint32_t k = 0; k < S; ++k)
						{
							if constexpr(Kp == CSRScatterOperation::PROD)
								{
									out[i * S + k] *= src[t * S + k];
								}
							else if constexpr(Kp == CSRScatterOperation::MIN)
								{
									out[i * S + k] = std::min<T1>(out[i * S + k], src[t * S + k]);
								}
							else if constexpr(Kp == CSRScatterOperation::MAX)
								{
									out[i * S + k] = std::max<T1>(out[i * S + k], src[t * S + k]);
								}
							else
								{
									out[i * S + k] += src[t * S + k];
								}
						}
				}

			if constexpr(Kp == CSRScatterOperation::MEAN)
				{
					const float scale = 1.0f / static_cast<float>(end - start);

					for(std::uint32_t k = 0; k < S; ++k)
						{
							out[i * S + k] *= scale;
						}
				}
			else if constexpr(Kp == CSRScatterOperation::VAR || Kp == CSRScatterOperation::STD)
				{
					const float scale = 1.0f / static_cast<float>(end - start);

					for(std::uint32_t k = 0; k < S; ++k)
						{
							mean_buffer[k] = static_cast<float>(out[i * S + k]) * scale;
							out[i * S + k] = 0;
						}

					for(std::uint32_t h = start; h < end; ++h)
						{
							const std::uint32_t t = tail[h];

							for(std::uint32_t k = 0; k < S; ++k)
								{
									const float diff = static_cast<float>(src[t * S + k]) - mean_buffer[k];
									out[i * S + k] += diff * diff;
								}
						}

					for(std::uint32_t k = 0; k < S; ++k)
						{
							if constexpr(Kp == CSRScatterOperation::VAR)
								{
									out[i * S + k] *= scale;
								}
							else
								{
									out[i * S + k] = std::sqrt(out[i * S + k] * scale);
								}
						}
				}
		}
}

}

#endif
