#include "operations.hpp"

namespace vext::cpu::details
{

enum class OperationKind
{
	NONE = 0,
	SUM,
	DIFF,
	MUL,
	MATMUL
};

template <OperationKind Tp>
void
operation_with_broadcast(
	float*                            dst,
	float*                            a,
	float*                            b,
	const std::uint64_t               N,
	const std::vector<std::uint64_t>& dims,
	const std::vector<std::uint64_t>& strides)
{
	const std::uint64_t dim_count = strides.size();

	std::vector<std::uint64_t> index;
	index.resize(dim_count, 0);

	std::uint64_t b_offset = 0;

	for(std::uint64_t i = 0; i < N; ++i)
		{
			if constexpr(Tp == OperationKind::SUM)
				{

					dst[i] = a[i] + b[b_offset];
				}
			else if constexpr(Tp == OperationKind::DIFF)
				{

					dst[i] = a[i] - b[b_offset];
				}
			if constexpr(Tp == OperationKind::MUL)
				{

					dst[i] = a[i] * b[b_offset];
				}

			for(std::uint64_t j = dim_count - 1; j >= 0; --j)
				{
					++index[j];

					b_offset += strides[j];

					if(index[j] < dims[j])
						{
							break;
						}

					index[j] = 0;
					b_offset = b_offset - dims[j] * strides[j];

					if(j == 0)
						{
							break;
						}
				}
		}
};

}

namespace vext::cpu
{

void
Operations::sum(
	float*              dst,
	float*              a,
	float*              b,
	const std::uint64_t N)
{
	for(std::uint64_t i = 0; i < N; ++i)
		{
			dst[i] = a[i] + b[i];
		}
}

void
Operations::sum(
	float*                            dst,
	float*                            a,
	float*                            b,
	const std::uint64_t               N,
	const std::vector<std::uint64_t>& dims,
	const std::vector<std::uint64_t>& strides)
{
	details::operation_with_broadcast<details::OperationKind::SUM>(dst, a, b, N, dims, strides);
}

void
Operations::diff(
	float*              dst,
	float*              a,
	float*              b,
	const std::uint64_t N)
{
	for(std::uint64_t i = 0; i < N; ++i)
		{
			dst[i] = a[i] - b[i];
		}
}

void
Operations::diff(
	float*                            dst,
	float*                            a,
	float*                            b,
	const std::uint64_t               N,
	const std::vector<std::uint64_t>& dims,
	const std::vector<std::uint64_t>& strides)
{
	details::operation_with_broadcast<details::OperationKind::DIFF>(dst, a, b, N, dims, strides);
}

void
Operations::mul(
	float*              dst,
	float*              a,
	float*              b,
	const std::uint64_t N)
{
	for(std::uint64_t i = 0; i < N; ++i)
		{
			dst[i] = a[i] * b[i];
		}
}

void
Operations::mul(
	float*                            dst,
	float*                            a,
	float*                            b,
	const std::uint64_t               N,
	const std::vector<std::uint64_t>& dims,
	const std::vector<std::uint64_t>& strides)
{
	details::operation_with_broadcast<details::OperationKind::MUL>(dst, a, b, N, dims, strides);
}

void
Operations::mul(
	float*              dst,
	float*              a,
	float*              b,
	const std::uint64_t M,
	const std::uint64_t P,
	const std::uint64_t N)
{
	for(std::uint64_t m = 0; m < M; ++m)
		{
			for(std::uint64_t p = 0; p < P; ++p)
				{
					const float a_value = a[m * P + p];

					for(std::uint64_t n = 0; n < N; ++n)
						{
							dst[m * N + n] += a_value * b[p * N + n];
						}
				}
		}
}

}
