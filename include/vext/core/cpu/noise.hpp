#ifndef __VEXT_CORE_CPU_NOISE_HPP__
#define __VEXT_CORE_CPU_NOISE_HPP__

#include <vext/type.hpp>

namespace vext::core::cpu
{

struct NoiseDescriptor
{
	std::uint32_t seed      = 0;
	std::uint32_t counter   = 0;
	std::int8_t   direction = 1;
};

inline NoiseDescriptor&
sequentional_noise_descriptor()
{
	static NoiseDescriptor instance{};
	return instance;
}

inline float
noise(
	const std::uint32_t x)
{
	const NoiseDescriptor& descriptor = sequentional_noise_descriptor();

	std::uint32_t h = descriptor.seed ^ (descriptor.counter * 0x85ebca6bu) ^ (x * 0x9e3779b9u);
	h ^= h >> 16;
	h *= 0x7feb352du;
	h ^= h >> 15;
	h *= 0x846ca68bu;
	h ^= h >> 16;

	return static_cast<float>(h >> 8) * (1.0f / 16777216.0f) * descriptor.direction;
}

}

#endif
