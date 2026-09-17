#ifndef __VEXT_CORE_CPU_OPS_ELEMENTWISE_UNARY_HPP__
#define __VEXT_CORE_CPU_OPS_ELEMENTWISE_UNARY_HPP__

#include <cmath>

#include <vext/core/cpu/noise.hpp>
#include <vext/core/type.hpp>
#include <vext/type.hpp>

namespace vext::core::cpu::ops
{

template <Op Kp, EvaluationMode Mp, typename T1, typename T2, core::Arithmetic... Is>
requires core::UnaryOperation<Kp>
static void
unary(
	T1*                       out,
	const T2*                 src,
	const std::uint32_t       N,
	const std::vector<float>& values)
{
	float a = 0.0f;
	float b = 0.0f;

	if(!values.empty())
		{
			if(values.size() == 2)
				{
					a = values[0];
					b = values[1];
				}
			else if(values.size() == 1)
				{
					a = values[0];
				}
		}

	T1 accumulate = 0;

	for(std::uint32_t i = 0; i < N; ++i)
		{
			if constexpr(Kp == Op::ABS)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = std::abs(src[i] + noise(i));
						}
					else
						{
							out[i] = std::abs(src[i]);
						}
				}
			else if constexpr(Kp == Op::SIN)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = std::sin(src[i] + noise(i));
						}
					else
						{
							out[i] = std::sin(src[i]);
						}
				}
			else if constexpr(Kp == Op::COS)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = std::cos(src[i] + noise(i));
						}
					else
						{
							out[i] = std::cos(src[i]);
						}
				}
			else if constexpr(Kp == Op::TANH)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = std::tanh(src[i] + noise(i));
						}
					else
						{
							out[i] = std::tanh(src[i]);
						}
				}
			else if constexpr(Kp == Op::NEG)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = -(src[i] + noise(i));
						}
					else
						{
							out[i] = -src[i];
						}
				}
			else if constexpr(Kp == Op::EXP)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = std::exp(src[i] + noise(i));
						}
					else
						{
							out[i] = std::exp(src[i]);
						}
				}
			else if constexpr(Kp == Op::LOG)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = std::log(src[i] + noise(i));
						}
					else
						{
							out[i] = std::log(src[i]);
						}
				}
			else if constexpr(Kp == Op::SQRT)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = std::sqrt(src[i] + noise(i));
						}
					else
						{
							out[i] = std::sqrt(src[i]);
						}
				}
			else if constexpr(Kp == Op::SQUARE)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = (src[i] + noise(i)) * (src[i] + noise(i));
						}
					else
						{
							out[i] = src[i] * src[i];
						}
				}
			else if constexpr(Kp == Op::ROUND)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = std::round(src[i] + noise(i));
						}
					else
						{
							out[i] = std::round(src[i]);
						}
				}
			else if constexpr(Kp == Op::SIGMOID)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = 1.0f / (1.0f + std::exp(-(src[i] + noise(i))));
						}
					else
						{
							out[i] = 1.0f / (1.0f + std::exp(-src[i]));
						}
				}
			else if constexpr(Kp == Op::SOFT_RELU)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = std::log(1.0f + std::exp(src[i] + noise(i)));
						}
					else
						{
							out[i] = std::log(1.0f + std::exp(src[i]));
						}
				}
			else if constexpr(Kp == Op::RELU)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							const auto value = src[i] + noise(i);
							out[i]           = value > 0 ? value : 0;
						}
					else
						{
							out[i] = src[i] > 0 ? src[i] : 0;
						}
				}
			else if constexpr(Kp == Op::SOFTMAX || Kp == Op::LOGSOFTMAX)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = std::exp(src[i] + noise(i));
						}
					else
						{
							out[i] = std::exp(src[i]);
						}
				}
			else if constexpr(Kp == Op::SOFTMIN)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = std::exp(-(src[i] + noise(i)));
						}
					else
						{
							out[i] = std::exp(-src[i]);
						}
				}
			else if constexpr(Kp == Op::LEAKY_RELU)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							const auto value = src[i] + noise(i);
							out[i]           = value > 0 ? value : (a * value);
						}
					else
						{
							out[i] = src[i] > 0 ? src[i] : (a * src[i]);
						}
				}
			else if constexpr(Kp == Op::ELU)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							const auto value = src[i] + noise(i);
							out[i]           = value > 0 ? value : a * (std::exp(value) - 1.0f);
						}
					else
						{
							out[i] = src[i] > 0 ? src[i] : a * (std::exp(src[i]) - 1.0f);
						}
				}
			else if constexpr(Kp == Op::SWISH)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							const auto value = src[i] + noise(i);
							out[i]           = value / (1.0f + std::exp(-a * value));
						}
					else
						{
							out[i] = src[i] / (1.0f + std::exp(-a * src[i]));
						}
				}
			else if constexpr(Kp == Op::LINEAR)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = a * (src[i] + noise(i)) + b;
						}
					else
						{
							out[i] = a * src[i] + b;
						}
				}
			else if constexpr(Kp == Op::CLIP)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = std::max(a, std::min(b, src[i] + noise(i)));
						}
					else
						{
							out[i] = std::max(a, std::min(b, src[i]));
						}
				}
			else if constexpr(Kp == Op::POW)
				{
					if constexpr(Mp == EvaluationMode::PERTURBED)
						{
							out[i] = a * std::pow(src[i] + noise(i), b);
						}
					else
						{
							out[i] = a * std::pow(src[i], b);
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
