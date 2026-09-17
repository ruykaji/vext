#ifndef __VEXT_CORE_CPU_OPS_REDUCTION_HPP__
#define __VEXT_CORE_CPU_OPS_REDUCTION_HPP__

#include <cmath>
#include <limits>
#include <vector>

#include <vext/core/cpu/noise.hpp>
#include <vext/core/type.hpp>
#include <vext/type.hpp>

namespace vext::core::cpu::ops
{

template <Op Kp, EvaluationMode Mp, typename T1, typename T2>
requires core::ReductionOperation<Kp>
void
reduce(
	T1* __restrict__ out,
	const T2* __restrict__ src,
	const std::uint32_t               N,
	const std::uint32_t               M,
	const std::vector<std::uint32_t>& keep_dims,
	const std::vector<std::uint32_t>& keep_strides,
	const std::vector<std::uint32_t>& reduce_dims,
	const std::vector<std::uint32_t>& reduce_strides)
{
	const std::uint64_t keep_size   = keep_dims.size();
	const std::uint64_t reduce_size = reduce_dims.size();

	std::vector<std::uint32_t> keep_coords(keep_dims.size(), 0);
	std::vector<std::uint32_t> reduce_coords(reduce_dims.size(), 0);

	std::uint32_t keep_offset = 0;

	for(std::uint32_t i = 0; i < N; ++i)
		{
			T1 accumulator = 0;

			if constexpr(Kp == Op::PROD)
				{
					accumulator = 1;
				}
			else if constexpr(Kp == Op::MIN)
				{
					accumulator = std::numeric_limits<T1>::max();
				}
			else if constexpr(Kp == Op::MAX)
				{
					accumulator = std::numeric_limits<T1>::lowest();
				}

			for(std::uint32_t j = 0; j < reduce_size; ++j)
				{
					reduce_coords[j] = 0;
				}

			std::uint32_t reduce_offset = 0;

			for(std::uint32_t j = 0; j < M; ++j)
				{
					if constexpr(Kp == Op::PROD)
						{
							if constexpr(Mp == EvaluationMode::PERTURBED)
								{
									const std::uint64_t index = keep_offset + reduce_offset;
									accumulator *= static_cast<T1>(src[index] + noise(index));
								}
							else
								{
									accumulator *= static_cast<T1>(src[keep_offset + reduce_offset]);
								}
						}
					else if constexpr(Kp == Op::MIN)
						{
							if constexpr(Mp == EvaluationMode::PERTURBED)
								{
									const std::uint64_t index = keep_offset + reduce_offset;
									accumulator               = std::min(accumulator, static_cast<T1>(src[index] + noise(index)));
								}
							else
								{
									accumulator = std::min(accumulator, static_cast<T1>(src[keep_offset + reduce_offset]));
								}
						}
					else if constexpr(Kp == Op::MAX)
						{
							if constexpr(Mp == EvaluationMode::PERTURBED)
								{
									const std::uint64_t index = keep_offset + reduce_offset;
									accumulator               = std::max(accumulator, static_cast<T1>(src[index] + noise(index)));
								}
							else
								{
									accumulator = std::max(accumulator, static_cast<T1>(src[keep_offset + reduce_offset]));
								}
						}
					else if constexpr(Kp == Op::L2_NORM)
						{
							if constexpr(Mp == EvaluationMode::PERTURBED)
								{
									const std::uint64_t index = keep_offset + reduce_offset;
									const T1            prod  = src[index] + noise(index);

									accumulator += prod * prod;
								}
							else
								{
									accumulator += src[keep_offset + reduce_offset] * src[keep_offset + reduce_offset];
								}
						}
					else
						{
							if constexpr(Mp == EvaluationMode::PERTURBED)
								{
									const std::uint64_t index = keep_offset + reduce_offset;
									accumulator += static_cast<T1>(src[index] + noise(index));
								}
							else
								{
									accumulator += static_cast<T1>(src[keep_offset + reduce_offset]);
								}
						}

					for(std::uint32_t k = reduce_size - 1;; --k)
						{
							++reduce_coords[k];

							if(reduce_coords[k] < reduce_dims[k])
								{
									reduce_offset += reduce_strides[k];
									break;
								}

							reduce_offset -= (reduce_dims[k] - 1) * reduce_strides[k];
							reduce_coords[k] = 0;

							if(k == 0)
								{
									break;
								}
						}
				}

			if constexpr(Kp == Op::VAR || Kp == Op::STD)
				{
					const float mean       = static_cast<float>(accumulator) / M;
					float       dispersion = 0.0f;

					for(std::uint32_t j = 0; j < reduce_size; ++j)
						{
							reduce_coords[j] = 0;
						}

					std::uint32_t reduce_offset = 0;

					for(std::uint32_t j = 0; j < M; ++j)
						{
							float diff = 0.0f;

							if constexpr(Mp == EvaluationMode::PERTURBED)
								{
									const std::uint64_t index = keep_offset + reduce_offset;
									diff                      = (src[index] + noise(index)) - mean;
								}
							else
								{
									diff = src[keep_offset + reduce_offset] - mean;
								}

							dispersion += diff * diff;

							for(std::uint32_t k = reduce_size - 1;; --k)
								{
									++reduce_coords[k];

									if(reduce_coords[k] < reduce_dims[k])
										{
											reduce_offset += reduce_strides[k];
											break;
										}

									reduce_offset -= (reduce_dims[k] - 1) * reduce_strides[k];
									reduce_coords[k] = 0;

									if(k == 0)
										{
											break;
										}
								}
						}

					if constexpr(Kp == Op::VAR)
						{
							out[i] = dispersion / M;
						}
					else
						{
							out[i] = std::sqrt(dispersion / M);
						}
				}
			else if constexpr(Kp == Op::MEAN)
				{
					out[i] = accumulator / M;
				}
			else if constexpr(Kp == Op::L2_NORM)
				{
					out[i] = std::sqrt(accumulator);
				}
			else
				{
					out[i] = accumulator;
				}

			for(std::uint32_t k = keep_size - 1;; --k)
				{
					++keep_coords[k];

					if(keep_coords[k] < keep_dims[k])
						{
							keep_offset += keep_strides[k];
							break;
						}

					keep_offset -= (keep_dims[k] - 1) * keep_strides[k];
					keep_coords[k] = 0;

					if(k == 0)
						{
							break;
						}
				}
		}
}

}

#endif
