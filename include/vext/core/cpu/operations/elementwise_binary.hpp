#ifndef __VEXT_CORE_CPU_OPERATIONS_ELEMENTWISE_BINARY_HPP__
#define __VEXT_CORE_CPU_OPERATIONS_ELEMENTWISE_BINARY_HPP__

#include <cmath>
#include <vector>

#include <vext/core/type.hpp>

namespace vext::core::cpu::operations
{

template <BinaryOperationKind Kp, typename T1, typename T2>
static void
binary(
	std::common_type_t<T1, T2>* out,
	const T1*                   a,
	const T2*                   b,
	const std::uint64_t         N)
{
	for(std::uint64_t i = 0; i < N; ++i)
		{
			if constexpr(Kp == BinaryOperationKind::ADD)
				{
					out[i] = a[i] + b[i];
				}
			else if constexpr(Kp == BinaryOperationKind::SUB)
				{
					out[i] = a[i] - b[i];
				}
			else if constexpr(Kp == BinaryOperationKind::MUL)
				{
					out[i] = a[i] * b[i];
				}
			else if constexpr(Kp == BinaryOperationKind::DIV)
				{
					out[i] = a[i] / b[i];
				}
			else if constexpr(Kp == BinaryOperationKind::POW)
				{
					out[i] = std::pow(a[i], b[i]);
				}
			else if constexpr(Kp == BinaryOperationKind::MIN)
				{
					out[i] = std::min(a[i], b[i]);
				}
			else if constexpr(Kp == BinaryOperationKind::MAX)
				{
					out[i] = std::max(a[i], b[i]);
				}
		}
}

template <BinaryOperationKind Kp, typename T1, typename T2>
static void
binary_with_broadcast(
	std::common_type_t<T1, T2>*       out,
	const T1*                         a,
	const T2*                         b,
	const std::uint64_t               N,
	const std::vector<std::uint64_t>& dims,
	const std::vector<std::uint64_t>& strides)
{
	const std::uint64_t dims_count = strides.size();

	std::vector<std::uint64_t> index;
	index.resize(dims_count, 0);

	std::uint64_t b_offset = 0;

	for(std::uint64_t i = 0; i < N; ++i)
		{
			if constexpr(Kp == BinaryOperationKind::ADD)
				{

					out[i] = a[i] + b[b_offset];
				}
			else if constexpr(Kp == BinaryOperationKind::SUB)
				{

					out[i] = a[i] - b[b_offset];
				}
			else if constexpr(Kp == BinaryOperationKind::MUL)
				{

					out[i] = a[i] * b[b_offset];
				}
			else if constexpr(Kp == BinaryOperationKind::DIV)
				{

					out[i] = a[i] / b[b_offset];
				}
			else if constexpr(Kp == BinaryOperationKind::POW)
				{

					out[i] = std::pow(a[i], b[b_offset]);
				}
			else if constexpr(Kp == BinaryOperationKind::MIN)
				{

					out[i] = std::min(a[i], b[b_offset]);
				}
			else if constexpr(Kp == BinaryOperationKind::MAX)
				{

					out[i] = std::max(a[i], b[b_offset]);
				}

			for(std::uint64_t j = dims_count - 1; j >= 0; --j)
				{
					++index[j];

					if(index[j] < dims[j])
						{
							b_offset += strides[j];
							break;
						}

					index[j] = 0;
					b_offset -= (dims[j] - 1) * strides[j];

					if(j == 0)
						{
							break;
						}
				}
		}
}

}

#endif
