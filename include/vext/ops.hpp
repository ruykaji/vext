#ifndef __VEXT_OPS_HPP__
#define __VEXT_OPS_HPP__

#include <vext/core/cpu/ops/csr_scatter.hpp>
#include <vext/core/cpu/ops/csr_spmv.hpp>
#include <vext/core/cpu/ops/elementwise_binary.hpp>
#include <vext/core/cpu/ops/elementwise_logical.hpp>
#include <vext/core/cpu/ops/elementwise_unary.hpp>
#include <vext/core/cpu/ops/linear_algebra.hpp>
#include <vext/core/cpu/ops/reduction.hpp>

#if VEXT_CUDA
#include <vext/core/cuda/ops/csr_scatter.cuh>
#include <vext/core/cuda/ops/csr_spmv.cuh>
#include <vext/core/cuda/ops/elementwise_binary.cuh>
#include <vext/core/cuda/ops/elementwise_logical.cuh>
#include <vext/core/cuda/ops/elementwise_unary.cuh>
#include <vext/core/cuda/ops/linear_algebra.cuh>
#include <vext/core/cuda/ops/reduction.cuh>
#endif

#include <vext/tensor.hpp>

namespace vext::ops
{

template <UnaryOp Kp, typename T1, Backend B1, core::Arithmetic... Is>
void
unary(
	Tensor<T1, B1>& tensor,
	Is... param)
{
	if constexpr(B1 == Backend::CPU)
		{
			core::cpu::ops::unary<Kp>(tensor.data(), tensor.length(), param...);
		}
	#if VEXT_CUDA
	else
		{
			core::cuda::ops::unary<Kp>(tensor.data(), tensor.length(), param...);
		}
	#else
	else
		{
			static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
		}
	#endif
}

template <BinaryOp Kp, typename T1, Backend B1, typename T2, Backend B2>
Tensor<std::common_type_t<T1, T2>, B1>
binary(
	const Tensor<T1, B1>& lhs,
	const Tensor<T2, B2>& rhs)
{
	static_assert(B1 == B2, "Error: Binary ops cannot be performed on tensors with different Backends!");

	Tensor<std::common_type_t<T1, T2>, B1> out(lhs.dims());

	if(lhs.dims() == rhs.dims())
		{
			if constexpr(B1 == Backend::CPU)
				{
					core::cpu::ops::binary<Kp>(out.data(), lhs.data(), rhs.data(), lhs.length());
				}
			#if VEXT_CUDA
			else
				{
					core::cuda::ops::binary<Kp>(out.data(), lhs.data(), rhs.data(), lhs.length());
				}
			#else
			else
				{
					static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
				}
			#endif
		}
	else
		{
			std::vector<std::uint32_t> strides;

			const std::uint64_t source_size = lhs.dims().size();
			const std::uint64_t target_size = rhs.dims().size();

			if(target_size == 1 && rhs.dims()[0] == 1)
				{
					strides = std::vector<std::uint32_t>(source_size, 0);
				}
			else
				{
					if(target_size > source_size)
						{
							throw std::runtime_error("Cannot perform binary operation on incompatible shapes.");
						}

					std::uint64_t offset_left = 0;
					std::uint64_t subset_size = 0;

					for(std::uint64_t i = 0; i < source_size; ++i)
						{
							if(lhs.dims()[i] == rhs.dims()[subset_size])
								{
									++subset_size;

									if(subset_size == target_size)
										{
											break;
										}

									continue;
								}

							subset_size = 0;
							offset_left = i + 1;
						}

					if(subset_size == 0)
						{
							throw std::runtime_error("Cannot perform binary operation on incompatible shapes.");
						}

					strides = lhs.strides();

					for(std::uint64_t i = 0; i < offset_left; ++i)
						{
							strides[i] = 0;
						}

					for(std::uint64_t i = offset_left + target_size; i < source_size; ++i)
						{
							strides[i] = 0;
						}
				}

			if constexpr(B1 == Backend::CPU)
				{
					core::cpu::ops::binary_with_broadcast<Kp>(out.data(), lhs.data(), rhs.data(), lhs.length(), lhs.dims(), strides);
				}
			#if VEXT_CUDA
			else
				{
					core::cuda::ops::binary_with_broadcast<Kp>(out.data(), lhs.data(), rhs.data(), lhs.length(), lhs.dims(), strides);
				}
			#else
			else
				{
					static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
				}
			#endif
		}

	return out;
}

template <LogicOp Kp, typename T1, Backend B1, typename T2, Backend B2>
Tensor<std::uint8_t, B1>
logical(
	const Tensor<T1, B1>& lhs,
	const Tensor<T2, B2>& rhs)
{
	static_assert(B1 == B2, "Error: Binary ops cannot be performed on tensors with different Backends!");

	Tensor<std::uint8_t, B1> out(lhs.dims());

	if(lhs.dims() != rhs.dims())
		{
			throw std::runtime_error("Cannot perform logical operation on incompatible shapes");
		}

	if constexpr(B1 == Backend::CPU)
		{
			core::cpu::ops::logical<Kp>(out.data(), lhs.data(), rhs.data(), out.length());
		}
	#if VEXT_CUDA
	else
		{
			core::cuda::ops::logical<Kp>(out.data(), lhs.data(), rhs.data(), out.length());
		}
	#else
	else
		{
			static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
		}
	#endif

	return out;
}

template <ReductionOp Kp, typename T1, Backend B1, std::integral... Is>
auto
reduction(
	const Tensor<T1, B1>& src,
	Is... axis)
{
	const std::uint32_t reduce_axis[] = { static_cast<std::uint32_t>(axis)... };
	const std::uint32_t reduce_size   = sizeof...(axis);

	const std::uint32_t dims_count = src.dims().size();

	std::vector<std::uint8_t> is_reduce_axis = {};
	std::uint32_t             keep_size      = 1;

	if constexpr(reduce_size != 0)
		{
			if(reduce_size > dims_count)
				{
					throw std::runtime_error("Cannot reduce tensor along non-existing axis.");
				}

			is_reduce_axis.resize(dims_count, 0);
			keep_size = dims_count - reduce_size;

			for(std::uint32_t i = 0; i < reduce_size; ++i)
				{
					std::uint32_t ax = reduce_axis[i];

					if(ax >= dims_count)
						{
							throw std::runtime_error("Cannot reduce tensor along non-existing axis.");
						}

					if(is_reduce_axis[ax])
						{
							throw std::runtime_error("Duplicate reduction axis.");
						}

					is_reduce_axis[ax] = 1;
				}
		}

	std::vector<std::uint32_t> keep_dims;
	std::vector<std::uint32_t> keep_strides;
	std::vector<std::uint32_t> reduce_dims;
	std::vector<std::uint32_t> reduce_strides;

	keep_dims.reserve(keep_size);
	keep_strides.reserve(keep_size);
	reduce_dims.reserve(reduce_size);
	reduce_strides.reserve(reduce_size);

	std::uint32_t N = 1;
	std::uint32_t M = 1;

	if constexpr(reduce_size == 0)
		{
			keep_dims.emplace_back(1);
			keep_strides.emplace_back(0);

			reduce_dims    = src.dims();
			reduce_strides = src.strides();
			M              = src.length();
		}
	else
		{
			for(std::uint32_t i = 0; i < dims_count; ++i)
				{
					if(is_reduce_axis[i])
						{
							reduce_dims.emplace_back(src.dims()[i]);
							reduce_strides.emplace_back(src.strides()[i]);
							M *= src.dims()[i];
						}
					else
						{
							keep_dims.emplace_back(src.dims()[i]);
							keep_strides.emplace_back(src.strides()[i]);
							N *= src.dims()[i];
						}
				}
		}

	Tensor<core::ReductionOut<Kp, T1>, B1> out(keep_dims);

	if constexpr(B1 == Backend::CPU)
		{
			core::cpu::ops::reduce<Kp>(out.data(), src.data(), N, M, keep_dims, keep_strides, reduce_dims, reduce_strides);
		}
	#if VEXT_CUDA
	else
		{
			core::cuda::ops::reduce<Kp>(out.data(), src.data(), N, M, keep_dims, keep_strides, reduce_dims, reduce_strides);
		}
	#else
	else
		{
			static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
		}
	#endif

	return out;
}

template <CSRScatterOp Kp, typename T1, Backend B1, Backend B2, Backend B3>
auto
csr_scatter(
	const Tensor<T1, B1>&            src,
	const Tensor<std::uint32_t, B2>& head,
	const Tensor<std::uint32_t, B3>& tail)
{
	static_assert(B1 == B2 && B1 == B3, "Error: CSR Scatter ops cannot be performed on tensors with different Backends!");

	if(src.dims()[0] != (head.dims()[0] - 1))
		{
			throw std::runtime_error("Cannot perform CSR Scatter operation on incompatible shapes");
		}

	Tensor<core::CSRScatterOut<Kp, T1>, B1> out(src.dims());

	if constexpr(B1 == Backend::CPU)
		{
			core::cpu::ops::csr_scatter<Kp>(out.data(), src.data(), head.data(), tail.data(), out.dims()[0], out.strides()[0]);
		}
	#if VEXT_CUDA
	else
		{
			core::cuda::ops::csr_scatter<Kp>(out.data(), src.data(), head.data(), tail.data(), out.dims()[0], out.strides()[0]);
		}
	#else
	else
		{
			static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
		}
	#endif

	return out;
}

template <CSRSpMVOp Kp, typename T1, Backend B1, typename T2, Backend B2, Backend B3, Backend B4>
auto
csr_spmv(
	const Tensor<T1, B1>&            A,
	const Tensor<std::uint32_t, B2>& head,
	const Tensor<std::uint32_t, B3>& tail,
	const Tensor<T2, B4>&            x)
{
	static_assert(B1 == B2 && B1 == B3 && B1 == B4, "Error: CSR Scatter ops cannot be performed on tensors with different Backends!");

    Tensor<core::CSRSpMVOut<Kp, T1>, B1> out(head.dims()[0] - 1);

	if constexpr(B1 == Backend::CPU)
		{
			core::cpu::ops::csr_spmv<Kp>(out.data(), A.data(), head.data(), tail.data(), x.data(), out.length());
		}
	#if VEXT_CUDA
	else
		{
			core::cuda::ops::csr_spmv<Kp>(out.data(), A.data(), head.data(), tail.data(), x.data(), out.length());
		}
	#else
	else
		{
			static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
		}
	#endif

	return out;
}

template <typename T1, Backend B1, typename T2, Backend B2>
Tensor<std::common_type_t<T1, T2>, B1>
matmul(
	const Tensor<T1, B1>& lhs,
	const Tensor<T2, B2>& rhs)
{
	static_assert(B1 == B2, "Error: Binary operations cannot be performed on tensors with different Backends!");

	const std::uint32_t lhs_shared = lhs.dims().back();
	const std::uint32_t rhs_shared = rhs.dims().front();

	if(lhs_shared != rhs_shared)
		{
			throw std::runtime_error("Cannot multiply tensors: left tensor's last dimension must match right tensor's first dimension.");
		}

	std::uint32_t lhs_combined = 1;
	std::uint32_t rhs_combined = 1;

	std::vector<std::uint32_t> remainder;
	remainder.reserve(lhs.dims().size() + rhs.dims().size());

	for(auto it = lhs.dims().begin(), end = std::prev(lhs.dims().end()); it != end; ++it)
		{
			const std::uint64_t dim = *it;

			lhs_combined *= dim;
			remainder.emplace_back(dim);
		}

	for(auto it = std::next(lhs.dims().begin()), end = lhs.dims().end(); it != end; ++it)
		{
			const std::uint64_t dim = *it;

			rhs_combined *= dim;
			remainder.emplace_back(dim);
		}

	if(remainder.empty())
		{
			remainder.emplace_back(1);
		}

	Tensor<std::common_type_t<T1, T2>, B1> out(remainder);

	if constexpr(B1 == Backend::CPU)
		{
			core::cpu::ops::matmul<T1, T2>(out.data(), lhs.data(), rhs.data(), lhs_combined, lhs_shared, rhs_combined);
		}
	#if VEXT_CUDA
	else
		{
			core::cuda::ops::matmul<T1, T2>(out.data(), lhs.data(), rhs.data(), lhs_combined, lhs_shared, rhs_combined);
		}
	#else
	else
		{
			static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
		}
	#endif

	return out;
}

}

#endif
