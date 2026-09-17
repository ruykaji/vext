#ifndef __VEXT_OPTIM_PARAMETER_HPP__
#define __VEXT_OPTIM_PARAMETER_HPP__

#include <algorithm>
#include <chrono>
#include <concepts>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <vext/core/cpu/noise.hpp>

#if VEXT_CUDA
#include <vext/core/cuda/noise.cuh>
#endif

#include <vext/tensor.hpp>

namespace vext::optim
{

template <typename Tp, Backend Bp, EvaluationMode Mp = EvaluationMode::PLAIN>
class Parameter
{
public:
	template <std::integral... Is>
	requires(std::same_as<Is, std::remove_cvref_t<Is>> && ...)
	Parameter(Is... dims)
		: __tensor(dims...),
		  __grad(dims...)
	{
	}

	Parameter(
		const std::vector<std::uint32_t>& dims)
		: __tensor(dims),
		  __grad(dims)
	{
	}

public:
	Tensor<Tp, Bp>&
	tensor() noexcept
	{
		return __tensor;
	}

	const Tensor<Tp, Bp>&
	tensor() const noexcept
	{
		return __tensor;
	}

public:
	Tensor<Tp, Bp>&
	grad() noexcept
	{
		return __grad;
	}

	const Tensor<Tp, Bp>&
	grad() const noexcept
	{
		return __grad;
	}

	void
	xavier_normal()
	{
		const std::vector<std::uint32_t>& shape = __tensor.dims();
		const auto [fan_in, fan_out]            = calculate_fan_in_and_fan_out(shape);
		const float sigma                       = 2.0f / (fan_in + fan_out);

		std::random_device       rd{};
		std::mt19937             gen{ rd() };
		std::normal_distribution d{ 0.0f, sigma };

		// clang-format off
        std::vector<float> values(__tensor.length(), 0);
        std::ranges::generate(values, [&](){ return d(gen); });
        __tensor.set_from(values);
		// clang-format on
	}

	void
	xavier_uniform()
	{
		const std::vector<std::uint32_t>& shape = __tensor.dims();
		const auto [fan_in, fan_out]            = calculate_fan_in_and_fan_out(shape);
		const float sigma                       = 2.0f / (fan_in + fan_out);
		const float a                           = std::sqrt(3.0f * sigma);

		std::random_device             rd{};
		std::mt19937                   gen{ rd() };
		std::uniform_real_distribution d{ -a, a };

		// clang-format off
        std::vector<float> values(__tensor.length(), 0);
        std::ranges::generate(values, [&](){ return d(gen); });
        __tensor.set_from(values);
		// clang-format on
	}

	void
	kaiming_normal(
		const float alph = 0.0f)
	{
		const std::vector<std::uint32_t>& shape = __tensor.dims();
		const auto [fan_in, _]                  = calculate_fan_in_and_fan_out(shape);
		const float gain                        = std::sqrt(2.0f / (1.0f + alph));
		const float sigma                       = gain * gain / fan_in;

		std::random_device       rd{};
		std::mt19937             gen{ rd() };
		std::normal_distribution d{ 0.0f, sigma };

		// clang-format off
        std::vector<float> values(__tensor.length(), 0);
        std::ranges::generate(values, [&](){ return d(gen); });
        __tensor.set_from(values);
		// clang-format on
	}

	void
	kaiming_uniform(
		const float alph = 0.0f)
	{
		const std::vector<std::uint32_t>& shape = __tensor.dims();
		const auto [fan_in, _]                  = calculate_fan_in_and_fan_out(shape);
		const float gain                        = std::sqrt(2.0f / (1.0f + alph));
		const float sigma                       = std::sqrt(3.0f / fan_in);
		const float a                           = gain * sigma;

		std::random_device             rd{};
		std::mt19937                   gen{ rd() };
		std::uniform_real_distribution d{ -a, a };

		// clang-format off
        std::vector<float> values(__tensor.length(), 0);
        std::ranges::generate(values, [&](){ return d(gen); });
        __tensor.set_from(values);
		// clang-format on
	}

private:
	static std::pair<std::uint64_t, std::uint64_t>
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

protected:
	Tensor<Tp, Bp> __tensor;
	Tensor<Tp, Bp> __grad;
};

template <typename Tp, Backend Bp>
class Parameter<Tp, Bp, EvaluationMode::PERTURBED> : public Parameter<Tp, Bp, EvaluationMode::PLAIN>
{
private:
	using Base = Parameter<Tp, Bp, EvaluationMode::PLAIN>;

public:
	using Base::Base;
	using Base::tensor;

	const Tensor<Tp, Bp>&
	tensor() const noexcept
	{
		if constexpr(Bp == Backend::CPU)
			{
				core::cpu::NoiseDescriptor& descriptor = core::cpu::sequentional_noise_descriptor();
				descriptor.seed                        = __seed[__itr];
				descriptor.counter                     = descriptor.counter + 1;
			}
		#if VEXT_CUDA
		else
			{
				core::cuda::NoiseDescriptor& descriptor = core::cuda::sequentional_noise_descriptor();
				descriptor.seed                         = __seed[__itr];
				descriptor.counter                      = descriptor.counter + 1;
			}
		#else
		else
			{
				static_assert(core::dependent_false<Bp>, "Unsupported backend");
			}
		#endif

		return Base::__tensor;
	}

public:
	void
	generate_seed(
		const std::uint64_t count)
	{
		std::mt19937                  gen(std::chrono::steady_clock::now().time_since_epoch().count());
		std::uniform_int_distribution distribution(1, std::numeric_limits<std::int32_t>::max());

		__seed.resize(count);

		for(std::uint64_t i = 0; i < count; ++i)
			{
				__seed[i] = distribution(gen);
			}
	}

	void
	set_seed(
		const std::vector<std::uint64_t>& seed)
	{
		if(seed.empty())
			{
				throw std::invalid_argument("A perturbed parameter requires at least one noise seed.");
			}

		__seed = seed;
		__itr  = 0;
	}

	void
	next_seed() noexcept
	{
		++__itr;

		if(__itr == __seed.size())
			{
				__itr = 0;
			}
	}

private:
	std::uint64_t              __itr  = 0;
	std::vector<std::uint64_t> __seed = { 0 };
};

}

#endif
