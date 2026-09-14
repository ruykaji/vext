#ifndef __VEXT_CORE_CPU_OPS_CSR_SCATTER_HPP__
#define __VEXT_CORE_CPU_OPS_CSR_SCATTER_HPP__

#include <algorithm>
#include <cmath>
#include <vector>

#include <vext/core/cpu/noise.hpp>
#include <vext/core/type.hpp>
#include <vext/type.hpp>

namespace vext::core::cpu::ops
{

template <Op Kp, ParameterMode Mp, typename T1, typename T2>
requires core::SparseReductionOperation<Kp>
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

	if constexpr(Kp == Op::VAR || Kp == Op::STD)
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
					if constexpr(Kp == Op::PROD)
						{
							out[i * S + k] = 1;
						}
					else if constexpr(Kp == Op::MIN)
						{
							out[i * S + k] = std::numeric_limits<T1>::max();
						}
					else if constexpr(Kp == Op::MAX)
						{
							out[i * S + k] = std::numeric_limits<T1>::lowest();
						}
					else
						{
							out[i * S + k] = 0;
						}
				}

			for(std::uint32_t h = start; h < end; ++h)
				{
					const std::uint32_t t = tail[h];

					for(std::uint32_t k = 0; k < S; ++k)
						{
							if constexpr(Kp == Op::PROD)
								{
									if constexpr(Mp == ParameterMode::PERTURBED)
										{
											const std::uint64_t index = t * S + k;
											out[i * S + k] *= src[index] + noise(index);
										}
									else
										{
											out[i * S + k] *= src[t * S + k];
										}
								}
							else if constexpr(Kp == Op::MIN)
								{
									if constexpr(Mp == ParameterMode::PERTURBED)
										{
											const std::uint64_t index = t * S + k;
											out[i * S + k]            = std::min<T1>(out[i * S + k], src[index] + noise(index));
										}
									else
										{
											out[i * S + k] = std::min<T1>(out[i * S + k], src[t * S + k]);
										}
								}
							else if constexpr(Kp == Op::MAX)
								{
									if constexpr(Mp == ParameterMode::PERTURBED)
										{
											const std::uint64_t index = t * S + k;
											out[i * S + k]            = std::max<T1>(out[i * S + k], src[index] + noise(index));
										}
									else
										{
											out[i * S + k] = std::max<T1>(out[i * S + k], src[t * S + k]);
										}
								}
							else
								{
									if constexpr(Mp == ParameterMode::PERTURBED)
										{
											const std::uint64_t index = t * S + k;
											out[i * S + k] += src[index] + noise(index);
										}
									else
										{
											out[i * S + k] += src[t * S + k];
										}
								}
						}
				}

			if constexpr(Kp == Op::MEAN)
				{
					const float scale = 1.0f / static_cast<float>(end - start);

					for(std::uint32_t k = 0; k < S; ++k)
						{
							out[i * S + k] *= scale;
						}
				}
			else if constexpr(Kp == Op::VAR || Kp == Op::STD)
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
									float diff = 0.0f;

									if constexpr(Mp == ParameterMode::PERTURBED)
										{
											const std::uint64_t index = t * S + k;
											diff                      = (static_cast<float>(src[t * S + k]) + noise(index)) - mean_buffer[k];
										}
									else
										{
											diff = static_cast<float>(src[t * S + k]) - mean_buffer[k];
										}

									out[i * S + k] += diff * diff;
								}
						}

					for(std::uint32_t k = 0; k < S; ++k)
						{
							if constexpr(Kp == Op::VAR)
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
