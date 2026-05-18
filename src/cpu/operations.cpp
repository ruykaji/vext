#include "operations.hpp"

namespace vext::cpu
{

void
Operations::sum(
	float*            dst,
	float*            a,
	float*            b,
	const std::size_t N)
{
	for(std::size_t i = 0; i < N; ++i)
		{
			dst[i] = a[i] + b[i];
		}
}

void
Operations::sum(
	float*            a,
	float*            b,
	const std::size_t N)
{
	for(std::size_t i = 0; i < N; ++i)
		{
			a[i] += b[i];
		}
}

void
Operations::diff(
	float*            dst,
	float*            a,
	float*            b,
	const std::size_t N)
{
	for(std::size_t i = 0; i < N; ++i)
		{
			dst[i] = a[i] - b[i];
		}
}

void
Operations::diff(
	float*            a,
	float*            b,
	const std::size_t N)
{
	for(std::size_t i = 0; i < N; ++i)
		{
			a[i] -= b[i];
		}
}

void
Operations::mul(
	float*            dst,
	float*            a,
	float*            b,
	const std::size_t M,
	const std::size_t P,
	const std::size_t N)
{
	for(std::size_t m = 0; m < M; ++m)
		{
			for(std::size_t p = 0; p < P; ++p)
				{
					const float a_value = a[m * P + p];

					for(std::size_t n = 0; n < N; ++n)
						{
							dst[m * N + n] += a_value * b[p * N + n];
						}
				}
		}
}

void
Operations::mul(
	float*            dst,
	float*            a,
	float*            b,
	const std::size_t N)
{
	for(std::size_t i = 0; i < N; ++i)
		{
			dst[i] = a[i] * b[i];
		}
}

void
Operations::mul(
	float*            a,
	float*            b,
	const std::size_t N)
{
	for(std::size_t i = 0; i < N; ++i)
		{
			a[i] *= b[i];
		}
}

}
