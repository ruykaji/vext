#ifndef __VEXT_OPTIM_PARAMETER_HPP__
#define __VEXT_OPTIM_PARAMETER_HPP__

#include <chrono>
#include <concepts>
#include <limits>
#include <random>
#include <unordered_map>
#include <vector>

#include <vext/tensor.hpp>

namespace vext::optim
{

template <typename T1, Backend B1, Noise Np = Noise::NONE>
class Parameter
{
	struct SeedItr
	{
		std::uint64_t itr    = 0;
		std::uint64_t offset = 0;
		std::uint64_t count  = 0;
	};

public:
	template <std::integral... Is>
	requires(std::same_as<Is, std::remove_cv_t<Is>> && ...)
	Parameter(
		Is... dims)
		: __tensor(dims...){};

public:
	operator Tensor<T1, B1>&() noexcept
	{
		return __tensor;
	}

	operator const Tensor<T1, B1>&() const noexcept
	{
		return __tensor;
	}

public:
	void
	generate_seed(
		const std::uint64_t count)
	{
		if(__seed.size() != count)
			{
				__seed.resize(count);
			}

		std::mt19937                  gen(std::chrono::steady_clock::now().time_since_epoch().count());
		std::uniform_int_distribution dis(1, std::numeric_limits<std::int32_t>::max());

		for(std::uint64_t i = 0; i < count; ++i)
			{
				__seed[i] = dis(gen);
			}
	}

	void
	next_seed(
		const std::uint64_t thread = 0)
	{
		constexpr std::uint64_t STEP = 1;

		const auto it = __itr.find(thread);

		if(it == __itr.end())
			{
				throw std::runtime_error("Cannot advance the seed iterator because no seed state is registered for the requested thread.");
			}

		auto& [itr, offset, count] = it->second;
		((itr + STEP >= count) ? (itr = 0) : (itr += STEP));
	}

private:
	Tensor<T1, B1> __tensor;

	/** Noise computaion related members */
	std::unordered_map<std::uint64_t, SeedItr> __itr  = {};
	std::vector<std::uint64_t>                 __seed = {};
};

}

#endif
