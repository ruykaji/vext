#ifndef __VEXT_CORE_CPU_OPS_ELEMENTWISE_UNARY_HPP__
#define __VEXT_CORE_CPU_OPS_ELEMENTWISE_UNARY_HPP__

#include <cmath>

#include <vext/core/type.hpp>
#include <vext/type.hpp>

namespace vext::core::cpu::ops
{

template <Op Kp, typename T1, core::Arithmetic... Is>
requires core::UnaryOperation<Kp>
static void
unary(
	T1* __restrict__ out,
	const std::uint32_t N,
	Is... param)
{
	T1 accumulate = 0;

	for(std::uint32_t i = 0; i < N; ++i)
		{
			if constexpr(Kp == Op::ABS)
				{
					out[i] = std::abs(out[i]);
				}
			else if constexpr(Kp == Op::SIN)
				{
					out[i] = std::sin(out[i]);
				}
			else if constexpr(Kp == Op::COS)
				{
					out[i] = std::cos(out[i]);
				}
			else if constexpr(Kp == Op::TANH)
				{
					out[i] = std::tanh(out[i]);
				}
			else if constexpr(Kp == Op::NEG)
				{
					out[i] = -out[i];
				}
			else if constexpr(Kp == Op::EXP)
				{
					out[i] = std::exp(out[i]);
				}
			else if constexpr(Kp == Op::LOG)
				{
					out[i] = std::log(out[i]);
				}
			else if constexpr(Kp == Op::SQRT)
				{
					out[i] = std::sqrt(out[i]);
				}
			else if constexpr(Kp == Op::SQUARE)
				{
					out[i] *= out[i];
				}
			else if constexpr(Kp == Op::ROUND)
				{
					out[i] = std::round(out[i]);
				}
			else if constexpr(Kp == Op::SIGMOID)
				{
					out[i] = 1.0f / (1.0f + std::exp(-out[i]));
				}
			else if constexpr(Kp == Op::SOFT_RELU)
				{
					out[i] = std::log(1.0f + std::exp(out[i]));
				}
			else if constexpr(Kp == Op::RELU)
				{
					out[i] = out[i] > 0 ? out[i] : 0;
				}
			else if constexpr(Kp == Op::SOFTMAX || Kp == Op::LOGSOFTMAX)
				{
					out[i] = std::exp(out[i]);
				}
			else if constexpr(Kp == Op::SOFTMIN)
				{
					out[i] = std::exp(-out[i]);
				}
			else if constexpr(Kp == Op::LEAKY_RELU)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					out[i]        = out[i] > 0 ? out[i] : (a * out[i]);
				}
			else if constexpr(Kp == Op::ELU)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					out[i]        = out[i] > 0 ? out[i] : a * (std::exp(out[i]) - 1.0f);
				}
			else if constexpr(Kp == Op::SWISH)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					out[i]        = out[i] / (1.0f + std::exp(-a * out[i]));
				}
			else if constexpr(Kp == Op::LINEAR)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					const float b = static_cast<float>(std::get<1>(std::tuple{ param... }));
					out[i]        = a * out[i] + b;
				}
			else if constexpr(Kp == Op::CLIP)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					const float b = static_cast<float>(std::get<1>(std::tuple{ param... }));
					out[i]        = std::max(a, std::min(b, out[i]));
				}
			else if constexpr(Kp == Op::POW)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					const float b = static_cast<float>(std::get<1>(std::tuple{ param... }));
					out[i]        = a * std::pow(out[i], b);
				}

			if constexpr(Kp == Op::SOFTMAX || Kp == Op::SOFTMIN || Kp == Op::LOGSOFTMAX)
				{
					accumulate += out[i];
				}
		}

	for(std::uint32_t i = 0; i < N; ++i)
		{
			if constexpr(Kp == Op::SOFTMAX || Kp == Op::SOFTMIN)
				{
					out[i] /= accumulate;
				}
			else if constexpr(Kp == Op::LOGSOFTMAX)
				{
					out[i] = std::log(out[i] / accumulate);
				}
		}
}

}

#endif
