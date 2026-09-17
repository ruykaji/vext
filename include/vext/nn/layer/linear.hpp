#ifndef __VEXT_LINEAR_HPP__
#define __VEXT_LINEAR_HPP__

#include <vext/nn/module.hpp>
#include <vext/ops.hpp>
#include <vext/optim/parameter.hpp>

namespace vext::nn::layer
{

template <Backend Bp, EvaluationMode Mp = EvaluationMode::PLAIN>
class Linear : public Module<Bp, Mp>
{
public:
	Linear(
		const std::uint64_t& input,
		const std::uint64_t& hidden_dim,
		const float          negative_slope = std::sqrt(5.0f))
		: Module<Bp, Mp>(__weight, __bias),
		  __weight(input, hidden_dim),
		  __bias(hidden_dim)
	{
		__weight.kaiming_uniform(negative_slope);
		__bias.kaiming_uniform();
	}

public:
	Tensor<float, Bp>
	operator()(
		const Tensor<float, Bp>& x) const
	{
		return binary<Op::ADD, Mp>(matmul<Mp>(x, __weight.tensor()), __bias.tensor());
	}

private:
	optim::Parameter<float, Bp, Mp> __weight;
	optim::Parameter<float, Bp, Mp> __bias;
};

}

#endif
