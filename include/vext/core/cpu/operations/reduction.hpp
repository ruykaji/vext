#ifndef __VEXT_CORE_CPU_OPERATIONS_REDUCTION_HPP__
#define __VEXT_CORE_CPU_OPERATIONS_REDUCTION_HPP__

#include <cmath>
#include <limits>
#include <vector>

#include <vext/core/type.hpp>

namespace vext::core::cpu::operations
{

template <ReductionOperation Kp, typename T1, typename T2>
void
reduce(
	T1*                               out,
	const T2*                         src,
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

			if constexpr(Kp == ReductionOperation::PROD)
				{
					accumulator = 1;
				}
			else if constexpr(Kp == ReductionOperation::MIN)
				{
					accumulator = std::numeric_limits<T1>::max();
				}
			else if constexpr(Kp == ReductionOperation::MAX)
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
					if constexpr(Kp == ReductionOperation::PROD)
						{
							accumulator *= static_cast<T1>(src[keep_offset + reduce_offset]);
						}
					else if constexpr(Kp == ReductionOperation::MIN)
						{
							accumulator = std::min(accumulator, static_cast<T1>(src[keep_offset + reduce_offset]));
						}
					else if constexpr(Kp == ReductionOperation::MAX)
						{
							accumulator = std::max(accumulator, static_cast<T1>(src[keep_offset + reduce_offset]));
						}
					else
						{
							accumulator += static_cast<T1>(src[keep_offset + reduce_offset]);
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

			if constexpr(Kp == ReductionOperation::VAR || Kp == ReductionOperation::STD)
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
							const float diff = src[keep_offset + reduce_offset] - mean;
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

					if constexpr(Kp == ReductionOperation::VAR)
						{
							out[i] = dispersion / M;
						}
					else
						{
							out[i] = std::sqrt(dispersion / M);
						}
				}
			else if constexpr(Kp == ReductionOperation::MEAN)
				{
					out[i] = accumulator / M;
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
