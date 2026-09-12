#ifndef __VEXT_LINEAR_HPP__
#define __VEXT_LINEAR_HPP__

#include <vext/nn/init.hpp>
#include <vext/nn/module.hpp>
#include <vext/ops.hpp>

namespace vext::nn::layer
{

template <Backend Bp>
class Linear : public Module<Bp>
{
	VEXT_MODULE(Bp);

public:
	Linear(
		const std::uint64_t& input,
		const std::uint64_t& hidden_dim,
		const float          negative_slope = std::sqrt(5.0f))
		: Module<Bp>(),
		  __weight(input, hidden_dim),
		  __bias(hidden_dim)
	{
		kaiming_uniform(__weight, negative_slope);
		kaiming_uniform(__bias);

		assign_parameter(&__weight);
		assign_parameter(&__bias);
	}

public:
	Tensor<float, Bp>
	operator()(
		const Tensor<float, Bp>& x) const
	{
		return binary<Op::ADD>(matmul(x, __weight), __bias);
	}

private:
	Tensor<float, Bp> __weight;
	Tensor<float, Bp> __bias;
};

}

#endif
