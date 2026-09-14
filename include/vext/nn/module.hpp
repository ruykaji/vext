#ifndef __VEXT_NN_MODULE_HPP__
#define __VEXT_NN_MODULE_HPP__

#include <stack>
#include <type_traits>

#include <vext/optim/parameter.hpp>
#include <vext/tensor.hpp>
#include <vext/type.hpp>

namespace vext::nn::module
{

template <typename Tp, Backend Bp, ParameterMode Mp>
class iterator
{
	using param_type      = optim::Parameter<float, Bp, Mp>;
	using vector_iterator = std::conditional_t<std::is_const_v<Tp>, typename std::vector<param_type*>::const_iterator, typename std::vector<param_type*>::iterator>;
	using module_pointer  = std::conditional_t<std::is_const_v<Tp>, Tp const*, Tp*>;

public:
	using iterator_category = std::input_iterator_tag;
	using value_type        = param_type;
	using difference_type   = std::ptrdiff_t;
	using pointer           = std::conditional_t<std::is_const_v<Tp>, param_type const*, param_type*>;
	using reference         = std::conditional_t<std::is_const_v<Tp>, param_type const&, param_type&>;

public:
	iterator() = default;

	iterator(
		module_pointer module)
	{
		if(module != nullptr)
			{
				__stack.emplace(module);
				next_module();
			}
	}

public:
	iterator&
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

	iterator
	operator++(int)
	{
		iterator tmp = *this;
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
		const iterator& lhs,
		const iterator& rhs)
	{
		if(lhs.__module == nullptr && rhs.__module == nullptr)
			{
				return true;
			}

		return lhs.__module == rhs.__module && lhs.__parameters_itr == rhs.__parameters_itr;
	}

	friend bool
	operator!=(
		const iterator& lhs,
		const iterator& rhs)
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

template <Backend Bp, ParameterMode Mp = ParameterMode::PLAIN>
class Module
{
	friend module::iterator<Module, Bp, Mp>;
	friend module::iterator<const Module, Bp, Mp>;

public:
	using iterator       = module::iterator<Module, Bp, Mp>;
	using const_iterator = module::iterator<const Module, Bp, Mp>;

public:
	template <typename... Args>
	Module(Args&... args)
	{
		// clang-format off
		([&]
        {
            using Tp = std::remove_cvref_t<Args>;

			if constexpr(std::derived_from<Tp, Module<Bp, Mp>>)
				{
					__modules.emplace_back(&args);
				}
			else if constexpr(std::is_same_v<Tp, optim::Parameter<float, Bp, Mp>>)
				{
					__parameters.emplace_back(&args);
				}
        }(), ...);
		// clang-format on
	}

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

private:
	std::vector<optim::Parameter<float, Bp, Mp>*> __parameters;
	std::vector<Module*>                          __modules;
};

}

#endif
