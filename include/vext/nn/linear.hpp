#ifndef __VEXT_LINEAR_HPP__
#define __VEXT_LINEAR_HPP__

#include <vext/nn/module.hpp>

namespace vext::nn
{

template <Backend Bp>
class Linear : public Module<Bp>
{
public:
	Linear(
		const std::uint64_t& input,
		const std::uint64_t& hidden_dim,
		const float          negative_slop = 0.0)
		: Module<Bp>(),
		  __weight(input, hidden_dim),
		  __bias(hidden_dim)
	{
		this->template add<InitializationKind::XAVIER_NORMAL>(&__weight);
		this->template add<InitializationKind::XAVIER_NORMAL>(&__bias);
	}

public:
	Tensor<float, Bp>
	operator()(
		const Tensor<float, Bp>& x) const
	{
		return x.matmul(__weight) + __bias;
	}

private:
	Tensor<float, Bp> __weight;
	Tensor<float, Bp> __bias;
};

}

#endif
