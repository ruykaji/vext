#ifndef __VEXT_CORE_CPU_OPS_ELEMENTWISE_UNARY_HPP__
#define __VEXT_CORE_CPU_OPS_ELEMENTWISE_UNARY_HPP__

#include <cmath>

#include <vext/core/type.hpp>
#include <vext/type.hpp>

namespace vext::core::cpu::ops
{

template <UnaryOp Kp, typename T1, core::Arithmetic... Is>
static void
unary(
	T1* __restrict__ out,
	const std::uint32_t N,
	Is... param)
{
	T1 accumulate = 0;

	for(std::uint32_t i = 0; i < N; ++i)
		{
			if constexpr(Kp == UnaryOp::ABS)
				{
					out[i] = std::abs(out[i]);
				}
			else if constexpr(Kp == UnaryOp::SIN)
				{
					out[i] = std::sin(out[i]);
				}
			else if constexpr(Kp == UnaryOp::COS)
				{
					out[i] = std::cos(out[i]);
				}
			else if constexpr(Kp == UnaryOp::TANH)
				{
					out[i] = std::tanh(out[i]);
				}
			else if constexpr(Kp == UnaryOp::NEG)
				{
					out[i] = -out[i];
				}
			else if constexpr(Kp == UnaryOp::EXP)
				{
					out[i] = std::exp(out[i]);
				}
			else if constexpr(Kp == UnaryOp::LOG)
				{
					out[i] = std::log(out[i]);
				}
			else if constexpr(Kp == UnaryOp::SQRT)
				{
					out[i] = std::sqrt(out[i]);
				}
			else if constexpr(Kp == UnaryOp::SQUARE)
				{
					out[i] *= out[i];
				}
			else if constexpr(Kp == UnaryOp::ROUND)
				{
					out[i] = std::round(out[i]);
				}
			else if constexpr(Kp == UnaryOp::SIGMOID)
				{
					out[i] = 1.0f / (1.0f + std::exp(-out[i]));
				}
			else if constexpr(Kp == UnaryOp::SOFT_RELU)
				{
					out[i] = std::log(1.0f + std::exp(out[i]));
				}
			else if constexpr(Kp == UnaryOp::RELU)
				{
					out[i] = out[i] > 0 ? out[i] : 0;
				}
			else if constexpr(Kp == UnaryOp::SOFTMAX || Kp == UnaryOp::LOGSOFTMAX)
				{
					out[i] = std::exp(out[i]);
				}
			else if constexpr(Kp == UnaryOp::SOFTMIN)
				{
					out[i] = std::exp(-out[i]);
				}
			else if constexpr(Kp == UnaryOp::LEAKY_RELU)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					out[i]        = out[i] > 0 ? out[i] : (a * out[i]);
				}
			else if constexpr(Kp == UnaryOp::ELU)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					out[i]        = out[i] > 0 ? out[i] : a * (std::exp(out[i]) - 1.0f);
				}
			else if constexpr(Kp == UnaryOp::SWISH)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					out[i]        = out[i] / (1.0f + std::exp(-a * out[i]));
				}
			else if constexpr(Kp == UnaryOp::LINEAR)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					const float b = static_cast<float>(std::get<1>(std::tuple{ param... }));
					out[i]        = a * out[i] + b;
				}
			else if constexpr(Kp == UnaryOp::CLIP)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					const float b = static_cast<float>(std::get<1>(std::tuple{ param... }));
					out[i]        = std::max(a, std::min(b, out[i]));
				}
			else if constexpr(Kp == UnaryOp::POW)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					const float b = static_cast<float>(std::get<1>(std::tuple{ param... }));
					out[i]        = a * std::pow(out[i], b);
				}

			if constexpr(Kp == UnaryOp::SOFTMAX || Kp == UnaryOp::SOFTMIN || Kp == UnaryOp::LOGSOFTMAX)
				{
					accumulate += out[i];
				}
		}

	for(std::uint32_t i = 0; i < N; ++i)
		{
			if constexpr(Kp == UnaryOp::SOFTMAX || Kp == UnaryOp::SOFTMIN)
				{
					out[i] /= accumulate;
				}
			else if constexpr(Kp == UnaryOp::LOGSOFTMAX)
				{
					out[i] = std::log(out[i] / accumulate);
				}
		}
}

}

#endif
