#ifndef __VEXT_OPERATOR_HPP__
#define __VEXT_OPERATOR_HPP__

#include <cstdint>
#include <vector>

namespace vext::cpu
{

class Operations
{
public:
	static void
	sum(
		float*              dst,
		float*              a,
		float*              b,
		const std::uint64_t N);

	static void
	sum(
		float*                            dst,
		float*                            a,
		float*                            b,
		const std::uint64_t               N,
		const std::vector<std::uint64_t>& dims,
		const std::vector<std::uint64_t>& strides);

	static void
	diff(
		float*              dst,
		float*              a,
		float*              b,
		const std::uint64_t N);

	static void
	diff(
		float*                            dst,
		float*                            a,
		float*                            b,
		const std::uint64_t               N,
		const std::vector<std::uint64_t>& dims,
		const std::vector<std::uint64_t>& strides);

	static void
	mul(
		float*              dst,
		float*              a,
		float*              b,
		const std::uint64_t N);

	static void
	mul(
		float*                            dst,
		float*                            a,
		float*                            b,
		const std::uint64_t               N,
		const std::vector<std::uint64_t>& dims,
		const std::vector<std::uint64_t>& strides);

	static void
	mul(
		float*              dst,
		float*              a,
		float*              b,
		const std::uint64_t M,
		const std::uint64_t P,
		const std::uint64_t N);
};

}

#endif
