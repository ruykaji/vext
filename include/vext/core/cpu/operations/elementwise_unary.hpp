#ifndef __VEXT_CORE_CPU_OPERATIONS_ELEMENTWISE_UNARY_HPP__
#define __VEXT_CORE_CPU_OPERATIONS_ELEMENTWISE_UNARY_HPP__

#include <cmath>

#include <vext/core/type.hpp>

namespace vext::core::cpu::operations
{

template <UnaryOperation Kp, typename T1, std::floating_point... Is>
static void
unary(
	T1* __restrict__ out,
	const std::uint32_t N,
	Is... param)
{
	T1 accumulate = 0;

	for(std::uint32_t i = 0; i < N; ++i)
		{
			if constexpr(Kp == UnaryOperation::ABS)
				{
					out[i] = std::abs(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::SIN)
				{
					out[i] = std::sin(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::COS)
				{
					out[i] = std::cos(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::TANH)
				{
					out[i] = std::tanh(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::NEG)
				{
					out[i] = -out[i];
				}
			else if constexpr(Kp == UnaryOperation::EXP)
				{
					out[i] = std::exp(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::LOG)
				{
					out[i] = std::log(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::SQRT)
				{
					out[i] = std::sqrt(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::SQUARE)
				{
					out[i] *= out[i];
				}
			else if constexpr(Kp == UnaryOperation::ROUND)
				{
					out[i] = std::round(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::SIGMOID)
				{
					out[i] = 1.0f / (1.0f + std::exp(-out[i]));
				}
			else if constexpr(Kp == UnaryOperation::SOFT_RELU)
				{
					out[i] = std::log(1.0f + std::exp(out[i]));
				}
			else if constexpr(Kp == UnaryOperation::RELU)
				{
					out[i] = out[i] > 0 ? out[i] : 0;
				}
			else if constexpr(Kp == UnaryOperation::SOFTMAX || Kp == UnaryOperation::LOGSOFTMAX)
				{
					out[i] = std::exp(out[i]);
				}
			else if constexpr(Kp == UnaryOperation::SOFTMIN)
				{
					out[i] = std::exp(-out[i]);
				}
			else if constexpr(Kp == UnaryOperation::LEAKY_RELU)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					out[i]        = out[i] > 0 ? out[i] : (a * out[i]);
				}
			else if constexpr(Kp == UnaryOperation::ELU)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					out[i]        = out[i] > 0 ? out[i] : a * (std::exp(out[i]) - 1.0f);
				}
			else if constexpr(Kp == UnaryOperation::SWISH)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					out[i]        = out[i] / (1.0f + std::exp(-a * out[i]));
				}
			else if constexpr(Kp == UnaryOperation::LINEAR)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					const float b = static_cast<float>(std::get<1>(std::tuple{ param... }));
					out[i]        = a * out[i] + b;
				}
			else if constexpr(Kp == UnaryOperation::CLIP)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					const float b = static_cast<float>(std::get<1>(std::tuple{ param... }));
					out[i]        = std::max(a, std::min(b, out[i]));
				}
			else if constexpr(Kp == UnaryOperation::POW)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					const float b = static_cast<float>(std::get<1>(std::tuple{ param... }));
					out[i]        = a * std::pow(out[i], b);
				}

			if constexpr(Kp == UnaryOperation::SOFTMAX || Kp == UnaryOperation::SOFTMIN || Kp == UnaryOperation::LOGSOFTMAX)
				{
					accumulate += out[i];
				}
		}

	for(std::uint32_t i = 0; i < N; ++i)
		{
			if constexpr(Kp == UnaryOperation::SOFTMAX || Kp == UnaryOperation::SOFTMIN)
				{
					out[i] /= accumulate;
				}
			else if constexpr(Kp == UnaryOperation::LOGSOFTMAX)
				{
					out[i] = std::log(out[i] / accumulate);
				}
		}
}

}

#endif
