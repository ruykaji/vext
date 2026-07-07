#ifndef __VEXT_NN_ACTIVATION_ELU_HPP__
#define __VEXT_NN_ACTIVATION_ELU_HPP__

#include <vext/nn/module.hpp>

namespace vext::nn::activation
{

template <Backend Bp, Mutation Mp = Mutation::IN_PLACE>
class ELU : public Module<Bp>
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
				x.elu(a);
			}
		else if constexpr(Mp == Mutation::COPY)
			{
				Tensor<float, Bp> copy(x);
				copy.elu(a);

				return copy;
			}
	}
};

}

#endif
