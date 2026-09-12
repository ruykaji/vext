#ifndef __VEXT_NN_ACTIVATION_LEAKY_RELU_HPP__
#define __VEXT_NN_ACTIVATION_LEAKY_RELU_HPP__

#include <vext/nn/module.hpp>
#include <vext/ops.hpp>

namespace vext::nn::activation
{

template <Backend Bp, Mutation Mp = Mutation::IN_PLACE>
class LeakyReLU : public Module<Bp>
{
	using input  = std::conditional_t<Mp == Mutation::IN_PLACE, Tensor<float, Bp>, const Tensor<float, Bp>>;
	using output = std::conditional_t<Mp == Mutation::IN_PLACE, void, Tensor<float, Bp>>;

public:
	output
	operator()(
		input&      x,
		const float a = 0.0f) const
	{
		if constexpr(Mp == Mutation::IN_PLACE)
			{
				unary<Op::LEAKY_RELU>(x, a);
			}
		else if constexpr(Mp == Mutation::COPY)
			{
				Tensor<float, Bp> copy(x);
				unary<Op::LEAKY_RELU>(copy, a);

				return copy;
			}
	}
};

}

#endif
