#ifndef __VEXT_LINEAR_HPP__
#define __VEXT_LINEAR_HPP__

#include <vext/nn/module.hpp>

namespace vext::nn
{

class Linear : public Module
{
public:
	Linear(
		const std::uint64_t&        input,
		const std::uint64_t&        hidden_dim,
		const type::ParameterInit parameter_init = type::ParameterInit::KAIMING_NORMAL,
		const float               negative_slop  = 0.0)
	{
		__weight = { { input, hidden_dim } };
		init(__weight, parameter_init, negative_slop);
		add("weight", &__weight);

		__bias = { { hidden_dim } };
		init(__bias, parameter_init, negative_slop);
		add("bias", &__bias);
	}

public:
	Tensor
	operator()(
		const Tensor& x)
	{
		return Tensor::matmul(x, __weight) + __bias;
	}

private:
	Tensor __weight = {};
	Tensor __bias   = {};
};

}

#endif
