#ifndef __VEXT_CORE_CPU_OPS_ELEMENTWISE_BINARY_HPP__
#define __VEXT_CORE_CPU_OPS_ELEMENTWISE_BINARY_HPP__

#include <cmath>
#include <vector>

#include <vext/core/cpu/noise.hpp>
#include <vext/core/type.hpp>
#include <vext/type.hpp>

namespace vext::core::cpu::ops
{

template <Op Kp, ParameterMode Mp, typename T1, typename T2, typename T3>
requires core::BinaryOperation<Kp>
static void
binary(
	T1*       out,
	const T2* a,
	const T3* __restrict__ b,
	const std::uint32_t N)
{
	for(std::uint32_t i = 0; i < N; ++i)
		{
			if constexpr(Kp == Op::ADD)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = a[i] + (b[i] + noise(i));
						}
					else
						{
							out[i] = a[i] + b[i];
						}
				}
			else if constexpr(Kp == Op::SUB)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = a[i] - (b[i] + noise(i));
						}
					else
						{
							out[i] = a[i] - b[i];
						}
				}
			else if constexpr(Kp == Op::MUL)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = a[i] * (b[i] + noise(i));
						}
					else
						{
							out[i] = a[i] * b[i];
						}
				}
			else if constexpr(Kp == Op::DIV)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = a[i] / (b[i] + noise(i));
						}
					else
						{
							out[i] = a[i] / b[i];
						}
				}
			else if constexpr(Kp == Op::POW)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = std::pow(a[i], (b[i] + noise(i)));
						}
					else
						{
							out[i] = std::pow(a[i], b[i]);
						}
				}
			else if constexpr(Kp == Op::MIN)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = std::min(a[i], (b[i] + noise(i)));
						}
					else
						{
							out[i] = std::min(a[i], b[i]);
						}
				}
			else if constexpr(Kp == Op::MAX)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = std::max(a[i], (b[i] + noise(i)));
						}
					else
						{
							out[i] = std::max(a[i], b[i]);
						}
				}
			else if constexpr(Kp == Op::PRELU)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = std::max<T1>(0, a[i]) + (b[i] + noise(i)) * std::min<T1>(0, a[i]);
						}
					else
						{
							out[i] = std::max<T1>(0, a[i]) + b[i] * std::min<T1>(0, a[i]);
						}
				}
		}
}

template <Op Kp, ParameterMode Mp, typename T1, typename T2, typename T3>
requires core::BinaryOperation<Kp>
static void
binary_with_broadcast(
	T1*                               out,
	const T2*                         a,
	const T3*                         b,
	const std::uint32_t               N,
	const std::vector<std::uint32_t>& dims,
	const std::vector<std::uint32_t>& strides)
{
	const std::uint32_t dims_count = strides.size();

	std::vector<std::uint32_t> index;
	index.resize(dims_count, 0);

	std::uint32_t b_offset = 0;

	for(std::uint32_t i = 0; i < N; ++i)
		{
			if constexpr(Kp == Op::ADD)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = a[i] + (b[b_offset] + noise(b_offset));
						}
					else
						{
							out[i] = a[i] + b[b_offset];
						}
				}
			else if constexpr(Kp == Op::SUB)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = a[i] - (b[b_offset] + noise(b_offset));
						}
					else
						{
							out[i] = a[i] - b[b_offset];
						}
				}
			else if constexpr(Kp == Op::MUL)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = a[i] * (b[b_offset] + noise(b_offset));
						}
					else
						{
							out[i] = a[i] * b[b_offset];
						}
				}
			else if constexpr(Kp == Op::DIV)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = a[i] / (b[b_offset] + noise(b_offset));
						}
					else
						{
							out[i] = a[i] / b[b_offset];
						}
				}
			else if constexpr(Kp == Op::POW)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = std::pow(a[i], (b[b_offset] + noise(b_offset)));
						}
					else
						{
							out[i] = std::pow(a[i], b[b_offset]);
						}
				}
			else if constexpr(Kp == Op::MIN)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = std::min(a[i], (b[b_offset] + noise(b_offset)));
						}
					else
						{
							out[i] = std::min(a[i], b[b_offset]);
						}
				}
			else if constexpr(Kp == Op::MAX)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = std::max(a[i], (b[b_offset] + noise(b_offset)));
						}
					else
						{
							out[i] = std::max(a[i], b[b_offset]);
						}
				}
			else if constexpr(Kp == Op::PRELU)
				{
					if constexpr(Mp == ParameterMode::PERTURBED)
						{
							out[i] = std::max<T1>(0, a[i]) + (b[b_offset] + noise(b_offset)) * std::min<T1>(0, a[i]);
						}
					else
						{
							out[i] = std::max<T1>(0, a[i]) + b[b_offset] * std::min<T1>(0, a[i]);
						}
				}
			for(std::uint32_t j = dims_count - 1;; --j)
				{
					++index[j];

					if(index[j] < dims[j])
						{
							b_offset += strides[j];
							break;
						}

					b_offset -= (dims[j] - 1) * strides[j];
					index[j] = 0;

					if(j == 0)
						{
							break;
						}
				}
		}
}

}

#endif
