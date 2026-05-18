#ifndef __VEXT_OPERATOR_HPP__
#define __VEXT_OPERATOR_HPP__

#include <cstdint>

namespace vext::cpu
{

class Operations
{
public:
	static void
	sum(
		float*            dst,
		float*            a,
		float*            b,
		const std::size_t N);

	static void
	sum(
		float*            a,
		float*            b,
		const std::size_t N);

	static void
	diff(
		float*            dst,
		float*            a,
		float*            b,
		const std::size_t N);

	static void
	diff(
		float*            a,
		float*            b,
		const std::size_t N);

	static void
	mul(
		float*            dst,
		float*            a,
		float*            b,
		const std::size_t M,
		const std::size_t P,
		const std::size_t N);

	static void
	mul(
		float*            dst,
		float*            a,
		float*            b,
		const std::size_t N);

	static void
	mul(
		float*            a,
		float*            b,
		const std::size_t N);
};

}

#endif
