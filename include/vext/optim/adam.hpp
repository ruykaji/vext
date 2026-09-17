#ifndef __VEXT_OPTIM_ADAM_HPP__
#define __VEXT_OPTIM_ADAM_HPP__

#include <vext/optim/parameter.hpp>

namespace vext::optim
{

template <Backend Bp, EvaluationMode Mp = EvaluationMode::PLAIN>
class Adam
{
	struct Dispatcher
	{
		Parameter<float, Bp, Mp>* param = nullptr;
		Tensor<float, Bp>         m;
		Tensor<float, Bp>         v;
	};

public:
	template <typename Ip>
	Adam(
		Ip                            begin,
		Ip                            end,
		const float                   alph  = 0.001,
		const std::pair<float, float> betas = { 0.9, 0.999 },
		const float                   eps   = 1e-9)
		: __alph(alph),
		  __bet1(betas.first),
		  __bet2(betas.second),
		  __eps(eps)
	{
		static_assert(std::is_class_v<Ip>, "");
		static_assert(std::is_same_v<typename Ip::value_type, Parameter<float, Bp, Mp>>, "");

		for(auto it = begin; it != end; ++it)
			{
				Parameter<float, Bp, Mp>&         param = *it;
				const std::vector<std::uint32_t>& dims  = param.tensor().dims();

				Dispatcher dispatcher{
					.param = &param,
					.m     = Tensor<float, Bp>(dims),
					.v     = Tensor<float, Bp>(dims)
				};
				__dispatchers.emplace(std::move(dispatcher));
			}
	}

public:
	void
	zero_grad()
	{
		for(const auto& [param, m, v] : __dispatchers)
			{
				assign(param->grad(), 0.0f);
			}
	}

	void
	step()
	{
		for(auto& [param, m, v] : __dispatchers)
			{
			}
	}

private:
	float __alph = 0.0f;
	float __bet1 = 0.0f;
	float __bet2 = 0.0f;
	float __eps  = 1e-9;

	std::vector<Dispatcher> __dispatchers;
};

}

#endif
