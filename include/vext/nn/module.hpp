#ifndef __VEXT_NN_MODULE_HPP__
#define __VEXT_NN_MODULE_HPP__

#include "vext/type.hpp"
#include <memory>
#include <random>
#include <stack>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <vext/tensor.hpp>

namespace vext::nn::module
{

template <typename Tp, bool IsConst = false>
class ParameterIterator
{
	using vector_iterator = std::conditional_t<IsConst, typename std::vector<Tensor*>::const_iterator, typename std::vector<Tensor*>::iterator>;

public:
	using iterator_category = std::input_iterator_tag;
	using value_type        = Tensor;
	using difference_type   = std::ptrdiff_t;
	using pointer           = std::conditional_t<IsConst, Tp const*, Tp*>;
	using reference         = std::conditional_t<IsConst, Tensor const&, Tensor&>;

public:
	ParameterIterator() = default;

	ParameterIterator(
		pointer module)
	{
		if(module != nullptr)
			{
				__stack.emplace(module);
				next_module();
			}
	}

public:
	ParameterIterator&
	operator++()
	{
		if(__module == nullptr)
			{
				return *this;
			}

		++__parameters_itr;

		if(__parameters_itr == __parameters_end)
			{
				next_module();
			}

		return *this;
	}

	ParameterIterator
	operator++(int)
	{
		ParameterIterator tmp = *this;
		++(*this);
		return tmp;
	}

	reference
	operator*() const noexcept
	{
		return **__parameters_itr;
	}

	friend bool
	operator==(
		const ParameterIterator& lhs,
		const ParameterIterator& rhs)
	{
		if(lhs.__module == nullptr && rhs.__module == nullptr)
			{
				return true;
			}

		return lhs.__module == rhs.__module && lhs.__parameters_itr == rhs.__parameters_itr;
	}

	friend bool
	operator!=(
		const ParameterIterator& lhs,
		const ParameterIterator& rhs)
	{
		return !(lhs == rhs);
	}

private:
	void
	next_module()
	{
		while(!__stack.empty())
			{
				pointer module = __stack.top();
				__stack.pop();

				for(auto child : module->__modules)
					{
						__stack.emplace(child);
					}

				__parameters_itr = module->__parameters.begin();
				__parameters_end = module->__parameters.end();

				if(__parameters_itr != __parameters_end)
					{
						__module = module;
						return;
					}
			}

		__module         = nullptr;
		__parameters_itr = {};
		__parameters_end = {};
	}

private:
	pointer             __module         = nullptr;
	vector_iterator     __parameters_itr = {};
	vector_iterator     __parameters_end = {};
	std::stack<pointer> __stack          = {};
};

}

namespace vext::nn
{

class Module
{
	friend module::ParameterIterator<Module>;
	friend module::ParameterIterator<Module, true>;

public:
	using iterator       = module::ParameterIterator<Module>;
	using const_iterator = module::ParameterIterator<Module, true>;

public:
	Module() = default;

	Module(
		const Module&) = delete;

	Module(
		Module&&) = delete;

	virtual ~Module() = default;

public:
	Module&
	operator=(
		const Module&) = delete;

	Module&
	operator=(
		Module&&) = delete;

public:
	iterator
	begin()
	{
		return { this };
	}

	iterator
	end()
	{
		return {};
	}

	const_iterator
	begin() const
	{
		return { this };
	}

	const_iterator
	end() const
	{
		return {};
	}

protected:
	void
	add(
		const std::string& name,
		Tensor*            param)
	{
		check_name_available(name);

		const std::uint64_t index = __parameters.size();

		__parameters.emplace_back(param);
		__name_to_index_parameters.emplace(name, index);
	}

	void
	add(
		const std::string& name,
		Module*            module)
	{
		if(module == nullptr)
			{
				throw std::invalid_argument("");
			}

		check_name_available(name);

		const std::uint64_t index = __modules.size();

		__modules.emplace_back(module);
		__name_to_index_modules.emplace(name, index);
	}

	void
	xavier_normal(
		Tensor& weight)
	{
		const Shape& shape           = weight.shape();
		const auto [fan_in, fan_out] = calculate_fan_in_and_fan_out(shape);
		const float sigma            = 2.0f / (fan_in + fan_out);

		std::random_device       rd{};
		std::mt19937             gen{ rd() };
		std::normal_distribution d{ 0.0f, sigma };

		for(std::uint64_t i = 0, end = shape.length(); i < end; ++i)
			{
				weight.data()[i] = d(gen);
			}
	}

	void
	init(
		Tensor&                   weight,
		const type::ParameterInit parameter_init = type::ParameterInit::XAVIER_NORMAL,
		const float               negative_slop  = 0.0f)
	{
		switch(parameter_init)
			{
				case type::ParameterInit::XAVIER_NORMAL :
					{
						xavier_normal(weight);
						break;
					}
				case type::ParameterInit::XAVIER_UNIFORM :
					{
						xavier_uniform(weight);
						break;
					}
				case type::ParameterInit::KAIMING_NORMAL :
					{
						kaiming_normal(weight, negative_slop);
						break;
					}
				case type::ParameterInit::KAIMING_UNIFORM :
					{
						kaiming_uniform(weight, negative_slop);
						break;
					}
			}
	}

private:
	void
	check_name_available(
		const std::string& name) const
	{
		if(__name_to_index_parameters.contains(name) || __name_to_index_modules.contains(name))
			{
				throw std::invalid_argument("");
			}
	}

	std::pair<std::uint64_t, std::uint64_t>
	calculate_fan_in_and_fan_out(
		const Shape& shape)
	{
		const std::uint64_t dims = shape.size();

		if(dims < 2)
			{
				throw std::runtime_error("");
			}

		const std::uint64_t input_fmaps  = shape[1];
		const std::uint64_t output_fmaps = shape[0];

		std::uint64_t receptive_field_size = 1;

		for(std::uint64_t i = 2; i < dims; ++i)
			{
				receptive_field_size *= shape[i];
			}

		const std::uint64_t fan_in  = input_fmaps * receptive_field_size;
		const std::uint64_t fan_out = output_fmaps * receptive_field_size;

		return { fan_in, fan_out };
	}

	void
	xavier_uniform(
		Tensor& weight)
	{
		const Shape& shape           = weight.shape();
		const auto [fan_in, fan_out] = calculate_fan_in_and_fan_out(shape);
		const float sigma            = 2.0f / (fan_in + fan_out);
		const float a                = std::sqrt(3.0f * sigma);

		std::random_device             rd{};
		std::mt19937                   gen{ rd() };
		std::uniform_real_distribution d{ -a, a };

		for(std::uint64_t i = 0, end = shape.length(); i < end; ++i)
			{
				weight[i] = d(gen);
			}
	}

	void
	kaiming_normal(
		Tensor&     weight,
		const float alph = 0.0f)
	{
		const Shape& shape     = weight.shape();
		const auto [fan_in, _] = calculate_fan_in_and_fan_out(shape);
		const float gain       = std::sqrt(2.0f / (1.0f + alph));
		const float sigma      = gain * gain / fan_in;

		std::random_device       rd{};
		std::mt19937             gen{ rd() };
		std::normal_distribution d{ 0.0f, sigma };

		for(std::uint64_t i = 0, end = shape.length(); i < end; ++i)
			{
				weight[i] = d(gen);
			}
	}

	void
	kaiming_uniform(
		Tensor&     weight,
		const float alph = 0.0f)
	{
		const Shape& shape     = weight.shape();
		const auto [fan_in, _] = calculate_fan_in_and_fan_out(shape);
		const float gain       = std::sqrt(2.0f / (1.0f + alph));
		const float sigma      = std::sqrt(3.0f / fan_in);
		const float a          = gain * sigma;

		std::random_device             rd{};
		std::mt19937                   gen{ rd() };
		std::uniform_real_distribution d{ -a, a };

		for(std::uint64_t i = 0, end = shape.length(); i < end; ++i)
			{
				weight[i] = d(gen);
			}
	}

private:
	std::vector<Tensor*>                           __parameters;
	std::unordered_map<std::string, std::uint64_t> __name_to_index_parameters;

	std::vector<Module*>                           __modules;
	std::unordered_map<std::string, std::uint64_t> __name_to_index_modules;
};

}

#endif
