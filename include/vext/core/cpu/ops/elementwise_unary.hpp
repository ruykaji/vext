#ifndef __VEXT_CORE_CPU_OPS_ELEMENTWISE_UNARY_HPP__
#define __VEXT_CORE_CPU_OPS_ELEMENTWISE_UNARY_HPP__

#include <cmath>

#include <vext/core/cpu/noise.hpp>
#include <vext/core/type.hpp>
#include <vext/type.hpp>

namespace vext::core::cpu::ops
{

template <Op Kp, ParameterMode Mp, typename T1, core::Arithmetic... Is>
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
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = std::abs(out[i] + noise(i));
						}
					else
						{
							out[i] = std::abs(out[i]);
						}
				}
			else if constexpr(Kp == Op::SIN)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = std::sin(out[i] + noise(i));
						}
					else
						{
							out[i] = std::sin(out[i]);
						}
				}
			else if constexpr(Kp == Op::COS)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = std::cos(out[i] + noise(i));
						}
					else
						{
							out[i] = std::cos(out[i]);
						}
				}
			else if constexpr(Kp == Op::TANH)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = std::tanh(out[i] + noise(i));
						}
					else
						{
							out[i] = std::tanh(out[i]);
						}
				}
			else if constexpr(Kp == Op::NEG)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = -(out[i] + noise(i));
						}
					else
						{
							out[i] = -out[i];
						}
				}
			else if constexpr(Kp == Op::EXP)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = std::exp(out[i] + noise(i));
						}
					else
						{
							out[i] = std::exp(out[i]);
						}
				}
			else if constexpr(Kp == Op::LOG)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = std::log(out[i] + noise(i));
						}
					else
						{
							out[i] = std::log(out[i]);
						}
				}
			else if constexpr(Kp == Op::SQRT)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = std::sqrt(out[i] + noise(i));
						}
					else
						{
							out[i] = std::sqrt(out[i]);
						}
				}
			else if constexpr(Kp == Op::SQUARE)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] *= (out[i] + noise(i));
						}
					else
						{
							out[i] *= out[i];
						}
				}
			else if constexpr(Kp == Op::ROUND)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = std::round(out[i] + noise(i));
						}
					else
						{
							out[i] = std::round(out[i]);
						}
				}
			else if constexpr(Kp == Op::SIGMOID)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = 1.0f / (1.0f + std::exp(-(out[i] + noise(i))));
						}
					else
						{
							out[i] = 1.0f / (1.0f + std::exp(-out[i]));
						}
				}
			else if constexpr(Kp == Op::SOFT_RELU)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = std::log(1.0f + std::exp(out[i] + noise(i)));
						}
					else
						{
							out[i] = std::log(1.0f + std::exp(out[i]));
						}
				}
			else if constexpr(Kp == Op::RELU)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] += noise(i);
						}

					out[i] = out[i] > 0 ? out[i] : 0;
				}
			else if constexpr(Kp == Op::SOFTMAX || Kp == Op::LOGSOFTMAX)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = std::exp(out[i] + noise(i));
						}
					else
						{
							out[i] = std::exp(out[i]);
						}
				}
			else if constexpr(Kp == Op::SOFTMIN)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = std::exp(-(out[i] + noise(i)));
						}
					else
						{
							out[i] = std::exp(-out[i]);
						}
				}
			else if constexpr(Kp == Op::LEAKY_RELU)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] += noise(i);
						}

					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					out[i]        = out[i] > 0 ? out[i] : (a * out[i]);
				}
			else if constexpr(Kp == Op::ELU)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));

					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] += noise(i);
						}

					out[i] = out[i] > 0 ? out[i] : a * (std::exp(out[i]) - 1.0f);
				}
			else if constexpr(Kp == Op::SWISH)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] += noise(i);
						}

					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					out[i]        = out[i] / (1.0f + std::exp(-a * out[i]));
				}
			else if constexpr(Kp == Op::LINEAR)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					const float b = static_cast<float>(std::get<1>(std::tuple{ param... }));

					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = a * (out[i] + noise(i)) + b;
						}
					else
						{
							out[i] = a * out[i] + b;
						}
				}
			else if constexpr(Kp == Op::CLIP)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					const float b = static_cast<float>(std::get<1>(std::tuple{ param... }));

					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = std::max(a, std::min(b, out[i] + noise(i)));
						}
					else
						{
							out[i] = std::max(a, std::min(b, out[i]));
						}
				}
			else if constexpr(Kp == Op::POW)
				{
					const float a = static_cast<float>(std::get<0>(std::tuple{ param... }));
					const float b = static_cast<float>(std::get<1>(std::tuple{ param... }));

					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = a * std::pow(out[i] + noise(i), b);
						}
					else
						{
							out[i] = a * std::pow(out[i], b);
						}
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
