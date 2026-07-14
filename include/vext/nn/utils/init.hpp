#ifndef __VEXT_NN_UTILS_INIT_HPP__
#define __VEXT_NN_UTILS_INIT_HPP__

#include <random>

#include <vext/tensor.hpp>

namespace vext::nn::utils
{

inline std::pair<std::uint64_t, std::uint64_t>
calculate_fan_in_and_fan_out(
	const std::vector<std::uint32_t>& shape) noexcept
{
	const std::uint64_t dims_count = shape.size();

	const std::uint64_t input_fmaps  = dims_count < 2 ? 1 : shape[1];
	const std::uint64_t output_fmaps = shape[0];

	std::uint64_t receptive_field_size = 1;

	for(std::uint64_t i = 2; i < dims_count; ++i)
		{
			receptive_field_size *= shape[i];
		}

	const std::uint64_t fan_in  = input_fmaps * receptive_field_size;
	const std::uint64_t fan_out = output_fmaps * receptive_field_size;

	return { fan_in, fan_out };
}

template <Backend Bp>
void
xavier_normal(
	Tensor<float, Bp>& weight)
{
	const std::vector<std::uint32_t>& shape = weight.shape();
	const auto [fan_in, fan_out]            = calculate_fan_in_and_fan_out(shape);
	const float sigma                       = 2.0f / (fan_in + fan_out);

	std::random_device       rd{};
	std::mt19937             gen{ rd() };
	std::normal_distribution d{ 0.0f, sigma };

	// clang-format off
	std::vector<float> values(weight.length(), 0);
    std::ranges::generate(values, [&](){ return d(gen); });
    weight.set_from(values);
	// clang-format on
}

template <Backend Bp>
void
xavier_uniform(
	Tensor<float, Bp>& weight)
{
	const std::vector<std::uint32_t>& shape = weight.shape();
	const auto [fan_in, fan_out]            = calculate_fan_in_and_fan_out(shape);
	const float sigma                       = 2.0f / (fan_in + fan_out);
	const float a                           = std::sqrt(3.0f * sigma);

	std::random_device             rd{};
	std::mt19937                   gen{ rd() };
	std::uniform_real_distribution d{ -a, a };

	// clang-format off
	std::vector<float> values(weight.length(), 0);
    std::ranges::generate(values, [&](){ return d(gen); });
    weight.set_from(values);
	// clang-format on
}

template <Backend Bp>
void
kaiming_normal(
	Tensor<float, Bp>& weight,
	const float        alph = 0.0f)
{
	const std::vector<std::uint32_t>& shape = weight.shape();
	const auto [fan_in, _]                  = calculate_fan_in_and_fan_out(shape);
	const float gain                        = std::sqrt(2.0f / (1.0f + alph));
	const float sigma                       = gain * gain / fan_in;

	std::random_device       rd{};
	std::mt19937             gen{ rd() };
	std::normal_distribution d{ 0.0f, sigma };

	// clang-format off
	std::vector<float> values(weight.length(), 0);
    std::ranges::generate(values, [&](){ return d(gen); });
    weight.set_from(values);
	// clang-format on
}

template <Backend Bp>
void
kaiming_uniform(
	Tensor<float, Bp>& weight,
	const float        alph = 0.0f)
{
	const std::vector<std::uint32_t>& shape = weight.shape();
	const auto [fan_in, _]                  = calculate_fan_in_and_fan_out(shape);
	const float gain                        = std::sqrt(2.0f / (1.0f + alph));
	const float sigma                       = std::sqrt(3.0f / fan_in);
	const float a                           = gain * sigma;

	std::random_device             rd{};
	std::mt19937                   gen{ rd() };
	std::uniform_real_distribution d{ -a, a };

	// clang-format off
	std::vector<float> values(weight.length(), 0);
    std::ranges::generate(values, [&](){ return d(gen); });
    weight.set_from(values);
	// clang-format on
}

}

#endif
