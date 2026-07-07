#ifndef __VEXT_NN_MODULE_HPP__
#define __VEXT_NN_MODULE_HPP__

#include <random>
#include <stack>

#include <vext/tensor.hpp>
#include <vext/type.hpp>

namespace vext::nn::module
{

template <typename Tp, Backend Bp>
class ParameterIterator
{
	using vector_iterator = std::conditional_t<std::is_const_v<Tp>, typename std::vector<Tensor<float, Bp>*>::const_iterator, typename std::vector<Tensor<float, Bp>*>::iterator>;
	using module_pointer  = std::conditional_t<std::is_const_v<Tp>, Tp const*, Tp*>;

public:
	using iterator_category = std::input_iterator_tag;
	using value_type        = Tensor<float, Bp>;
	using difference_type   = std::ptrdiff_t;
	using pointer           = std::conditional_t<std::is_const_v<Tp>, Tensor<float, Bp> const*, Tensor<float, Bp>*>;
	using reference         = std::conditional_t<std::is_const_v<Tp>, Tensor<float, Bp> const&, Tensor<float, Bp>&>;

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
	friend module::ParameterIterator<Module, Bp>;
	friend module::ParameterIterator<const Module, Bp>;

public:
	using iterator       = module::ParameterIterator<Module, Bp>;
	using const_iterator = module::ParameterIterator<const Module, Bp>;

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
	void
	assign_parameter(
		Tensor<float, Bp>* tensor)
	{
		__parameters.emplace_back(tensor);
	}

	void
	assign_module(
		Module* module)
	{
		__modules.emplace_back(module);
	}

private:
	std::vector<Tensor<float, Bp>*> __parameters;
	std::vector<Module*>            __modules;
};

}

#define VEXT_MODULE(param)                          \
	using vext::nn::Module<param>::assign_parameter; \
	using vext::nn::Module<param>::assign_module;

#endif
