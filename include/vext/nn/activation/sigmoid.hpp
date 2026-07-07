#ifndef __VEXT_NN_ACTIVATION_SIGMOID_HPP__
#define __VEXT_NN_ACTIVATION_SIGMOID_HPP__

#include <vext/nn/module.hpp>

namespace vext::nn::activation
{

template <Backend Bp, Mutation Mp = Mutation::IN_PLACE>
class Sigmoid : public Module<Bp>
{
	using input  = std::conditional_t<Mp == Mutation::IN_PLACE, Tensor<float, Bp>, const Tensor<float, Bp>>;
	using output = std::conditional_t<Mp == Mutation::IN_PLACE, void, Tensor<float, Bp>>;

public:
	output
	operator()(
		input& x) const
	{
		if constexpr(Mp == Mutation::IN_PLACE)
			{
				x.sigmoid();
			}
		else if constexpr(Mp == Mutation::COPY)
			{
				Tensor<float, Bp> copy(x);
				copy.sigmoid();

                return copy;
			}
	}
};

}

#endif
