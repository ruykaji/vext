#ifndef __VEXT_NN_MODULE_HPP__
#define __VEXT_NN_MODULE_HPP__

#include <random>
#include <stack>

#include <vext/tensor.hpp>
#include <vext/type.hpp>

namespace vext::nn::module
{

template <typename Tp>
class ParameterIterator
{
	using vector_iterator = std::conditional_t<std::is_const_v<Tp>, typename std::vector<Tensor<float>*>::const_iterator, typename std::vector<Tensor<float>*>::iterator>;
	using module_pointer  = std::conditional_t<std::is_const_v<Tp>, Tp const*, Tp*>;

public:
	using iterator_category = std::input_iterator_tag;
	using value_type        = Tensor<float>;
	using difference_type   = std::ptrdiff_t;
	using pointer           = std::conditional_t<std::is_const_v<Tp>, Tensor<float> const*, Tensor<float>*>;
	using reference         = std::conditional_t<std::is_const_v<Tp>, Tensor<float> const&, Tensor<float>&>;

public:
	ParameterIterator() = default;

	ParameterIterator(
		module_pointer module)
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

	pointer
	operator->() const noexcept
	{
		return *__parameters_itr;
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
				module_pointer module = __stack.top();
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
	module_pointer             __module         = nullptr;
	vector_iterator            __parameters_itr = {};
	vector_iterator            __parameters_end = {};
	std::stack<module_pointer> __stack          = {};
};

}

namespace vext::nn
{

template <Backend Bp>
class Module
{
	friend module::ParameterIterator<Module>;
	friend module::ParameterIterator<const Module>;

public:
	using iterator       = module::ParameterIterator<Module>;
	using const_iterator = module::ParameterIterator<const Module>;

public:
	virtual ~Module() = default;

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
	template <Initialization Kp>
	void
	add(
		Tensor<float, Bp>* param,
		const float        alph = 0.0f)
	{
		Tensor<float, Bp>& ref = *__parameters.emplace_back(param);

		if constexpr(Kp == Initialization::XAVIER_NORMAL)
			{
				xavier_normal(ref);
			}
		else if constexpr(Kp == Initialization::XAVIER_UNIFORM)
			{
				xavier_uniform(ref);
			}
		else if constexpr(Kp == Initialization::KAIMING_NORMAL)
			{
				kaiming_normal(ref, alph);
			}
		else if constexpr(Kp == Initialization::KAIMING_UNIFORM)
			{
				kaiming_uniform(ref, alph);
			}
	}

	void
	add(
		Module* module)
	{
		__modules.emplace_back(module);
	}

private:
	static std::pair<std::uint64_t, std::uint64_t>
	calculate_fan_in_and_fan_out(
		const Shape& shape) noexcept
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

	static void
	xavier_normal(
		Tensor<float, Bp>& weight)
	{
		const Shape& shape           = weight.shape();
		const auto [fan_in, fan_out] = calculate_fan_in_and_fan_out(shape);
		const float sigma            = 2.0f / (fan_in + fan_out);

		std::random_device       rd{};
		std::mt19937             gen{ rd() };
		std::normal_distribution d{ 0.0f, sigma };

		for(std::uint64_t i = 0, end = shape.length(); i < end; ++i)
			{
				weight.item(i) = d(gen);
			}
	}

	static void
	xavier_uniform(
		Tensor<float, Bp>& weight)
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
				weight.item(i) = d(gen);
			}
	}

	static void
	kaiming_normal(
		Tensor<float, Bp>& weight,
		const float        alph = 0.0f)
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
				weight.item(i) = d(gen);
			}
	}

	static void
	kaiming_uniform(
		Tensor<float, Bp>& weight,
		const float        alph = 0.0f)
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
				weight.item(i) = d(gen);
			}
	}

private:
	std::vector<Tensor<float, Bp>*> __parameters;
	std::vector<Module*>            __modules;
};

}

#define VEXT_MODULE(param) \
	using vext::nn::Module<param>::add;

#endif
