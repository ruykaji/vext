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

namespace vext
{

inline Axes
axes(
	std::initializer_list<std::int32_t> values)
{
	return Axes(values);
}

template <Op Kp, typename T1, Backend B1, core::Arithmetic... Is>
requires core::UnaryOperation<Kp>
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

template <Op Kp, typename T1, Backend B1, typename T2, Backend B2, typename To = core::no_value_t>
requires core::BinaryOperation<Kp>
auto
binary(
	const Tensor<T1, B1>& lhs,
	const Tensor<T2, B2>& rhs,
	To&&                  maybe_out = {})
{
	constexpr bool IS_OUT_DEFINED = !std::is_same_v<To, core::no_value_t>;

	if constexpr(IS_OUT_DEFINED)
		{
			constexpr bool IS_MUTABLE = !std::is_const_v<std::remove_reference_t<To>>;
			static_assert(IS_MUTABLE, "");

			constexpr bool IS_TENSOR_INSTANTIATION = core::is_tensor_instantiation<std::remove_reference_t<To>, Tensor>::value;
			static_assert(IS_TENSOR_INSTANTIATION, "");
		}

	using CommonType = std::common_type_t<T1, T2>;
	using TensorOut  = std::conditional_t<IS_OUT_DEFINED, To, Tensor<CommonType, B1>>;

	constexpr bool IS_SAME_DEVICE = (B1 == B2 && B1 == (std::remove_reference_t<TensorOut>::backend_type));
	static_assert(IS_SAME_DEVICE, "Error: Binary ops cannot be performed on tensors with different Backends!");

	const auto assign_out = [&]() -> TensorOut
		{
			if constexpr(IS_OUT_DEFINED)
				{
					return maybe_out;
				}
			else
				{
					return TensorOut(lhs.dims());
				}
		};

	TensorOut out = assign_out();

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
							throw std::runtime_error("Binary operation cannot broadcast the right-hand tensor shape to the left-hand tensor shape.");
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
							throw std::runtime_error("Binary operation cannot broadcast the right-hand tensor shape to the left-hand tensor shape.");
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

	if constexpr(IS_OUT_DEFINED)
		{
			return;
		}
	else
		{
			return out;
		}
}

template <Op Kp, typename T1, Backend B1, typename T2, Backend B2, typename To = core::no_value_t>
requires core::LogicalOperation<Kp>
auto
logical(
	const Tensor<T1, B1>& lhs,
	const Tensor<T2, B2>& rhs,
	To&&                  maybe_out = {})
{
	constexpr bool IS_OUT_DEFINED = !std::is_same_v<To, core::no_value_t>;

	if constexpr(IS_OUT_DEFINED)
		{
			constexpr bool IS_MUTABLE = !std::is_const_v<std::remove_reference_t<To>>;
			static_assert(IS_MUTABLE, "");

			constexpr bool IS_TENSOR_INSTANTIATION = core::is_tensor_instantiation<std::remove_cvref_t<To>, Tensor>::value;
			static_assert(IS_TENSOR_INSTANTIATION, "");
		}

	using CommonType = std::uint8_t;
	using TensorOut  = std::conditional_t<IS_OUT_DEFINED, To, Tensor<CommonType, B1>>;

	constexpr bool IS_SAME_DEVICE = (B1 == B2 && B1 == (std::remove_reference_t<TensorOut>::backend_type));
	static_assert(IS_SAME_DEVICE, "Error: Binary ops cannot be performed on tensors with different Backends!");

	const auto assign_out = [&]() -> TensorOut
		{
			if constexpr(IS_OUT_DEFINED)
				{
					return maybe_out;
				}
			else
				{
					return TensorOut(lhs.dims());
				}
		};

	TensorOut out = assign_out();

	if(lhs.dims() != rhs.dims())
		{
			throw std::runtime_error("Logical operation requires both tensors to have identical shapes.");
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

	if constexpr(IS_OUT_DEFINED)
		{
			return;
		}
	else
		{
			return out;
		}
}

template <Op Kp, typename T1, Backend B1, typename Ta = core::no_value_t, typename To = core::no_value_t>
requires core::ReductionOperation<Kp>
auto
reduction(
	const Tensor<T1, B1>& src,
	Ta&&                  axis      = {},
	To&&                  maybe_out = {})
{
	constexpr bool IS_OUT_DEFINED = !std::is_same_v<To, core::no_value_t>;

	if constexpr(IS_OUT_DEFINED)
		{
			constexpr bool IS_MUTABLE = !std::is_const_v<std::remove_reference_t<To>>;
			static_assert(IS_MUTABLE, "");

			constexpr bool IS_TENSOR_INSTANTIATION = core::is_tensor_instantiation<std::remove_cvref_t<To>, Tensor>::value;
			static_assert(IS_TENSOR_INSTANTIATION, "");
		}

	using CommonType = core::ReductionOut<Kp, T1>;
	using TensorOut  = std::conditional_t<IS_OUT_DEFINED, To, Tensor<CommonType, B1>>;

	constexpr bool IS_SAME_DEVICE = (B1 == (std::remove_reference_t<TensorOut>::backend_type));
	static_assert(IS_SAME_DEVICE, "Error: Binary ops cannot be performed on tensors with different Backends!");

	constexpr bool IS_REDUCE_AXIS = !std::is_same_v<Ta, core::no_value_t>;

	if constexpr(IS_REDUCE_AXIS)
		{
			constexpr bool IS_AXES = std::is_same_v<std::remove_cvref_t<Ta>, Axes>;
			static_assert(IS_AXES, "Reduction axes must be provided as vext::Axes.");
		}

	const std::uint32_t dims_count = src.dims().size();

	std::vector<std::uint8_t> is_reduce_axis = {};
	std::uint32_t             keep_size      = 1;
	std::uint32_t             reduce_count   = 0;

	if constexpr(IS_REDUCE_AXIS)
		{
			reduce_count = axis.size();

			if(reduce_count > dims_count)
				{
					throw std::runtime_error("Reduction specifies more axes than the tensor rank.");
				}

			is_reduce_axis.resize(dims_count, 0);
			keep_size = dims_count - reduce_count;

			for(const auto ax : axis)
				{
					if(ax < 0)
						{
							throw std::runtime_error("Reduction axis cannot be negative.");
						}

					if(static_cast<std::uint32_t>(ax) >= dims_count)
						{
							throw std::runtime_error("Reduction axis is outside the tensor rank.");
						}

					if(is_reduce_axis[ax])
						{
							throw std::runtime_error("Reduction axes must be unique; a duplicate axis was provided.");
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

	std::uint32_t N = 1;
	std::uint32_t M = 1;

	if constexpr(IS_REDUCE_AXIS)
		{
			reduce_dims.reserve(reduce_count);
			reduce_strides.reserve(reduce_count);

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
	else
		{
			keep_dims.emplace_back(1);
			keep_strides.emplace_back(0);

			reduce_dims    = src.dims();
			reduce_strides = src.strides();
			M              = src.length();
		}

	const auto assign_out = [&]() -> TensorOut
		{
			if constexpr(IS_OUT_DEFINED)
				{
					return maybe_out;
				}
			else
				{
					return TensorOut(keep_dims);
				}
		};

	TensorOut out = assign_out();

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

	if constexpr(IS_OUT_DEFINED)
		{
			return;
		}
	else
		{
			return out;
		}
}

template <Op Kp, typename T1, Backend B1, Backend B2, Backend B3, typename To = core::no_value_t>
requires core::SparseReductionOperation<Kp>
auto
csr_scatter(
	const Tensor<T1, B1>&            src,
	const Tensor<std::uint32_t, B2>& head,
	const Tensor<std::uint32_t, B3>& tail,
	To&&                             maybe_out = {})
{
	constexpr bool IS_OUT_DEFINED = !std::is_same_v<To, core::no_value_t>;

	if constexpr(IS_OUT_DEFINED)
		{
			constexpr bool IS_MUTABLE = !std::is_const_v<std::remove_reference_t<To>>;
			static_assert(IS_MUTABLE, "");

			constexpr bool IS_TENSOR_INSTANTIATION = core::is_tensor_instantiation<std::remove_cvref_t<To>, Tensor>::value;
			static_assert(IS_TENSOR_INSTANTIATION, "");
		}

	using CommonType = core::CSRScatterOut<Kp, T1>;
	using TensorOut  = std::conditional_t<IS_OUT_DEFINED, To, Tensor<CommonType, B1>>;

	constexpr bool IS_SAME_DEVICE = (B1 == B2 && B1 == B3 && B1 == (std::remove_reference_t<TensorOut>::backend_type));
	static_assert(IS_SAME_DEVICE, "Error: Binary ops cannot be performed on tensors with different Backends!");

	if(src.dims()[0] != (head.dims()[0] - 1))
		{
			throw std::runtime_error("CSR scatter requires the source row count to equal the head tensor length minus one.");
		}

	const auto assign_out = [&]() -> TensorOut
		{
			if constexpr(IS_OUT_DEFINED)
				{
					return maybe_out;
				}
			else
				{
					return TensorOut(src.dims());
				}
		};

	TensorOut out = assign_out();

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

	if constexpr(IS_OUT_DEFINED)
		{
			return;
		}
	else
		{
			return out;
		}
}

template <Op Kp, typename T1, Backend B1, typename T2, Backend B2, Backend B3, Backend B4, typename To = core::no_value_t>
requires core::SparseReductionOperation<Kp>
auto
csr_spmv(
	const Tensor<T1, B1>&            A,
	const Tensor<std::uint32_t, B2>& head,
	const Tensor<std::uint32_t, B3>& tail,
	const Tensor<T2, B4>&            x,
	To&&                             maybe_out = {})
{
	constexpr bool IS_OUT_DEFINED = !std::is_same_v<To, core::no_value_t>;

	if constexpr(IS_OUT_DEFINED)
		{
			constexpr bool IS_MUTABLE = !std::is_const_v<std::remove_reference_t<To>>;
			static_assert(IS_MUTABLE, "");

			constexpr bool IS_TENSOR_INSTANTIATION = core::is_tensor_instantiation<std::remove_cvref_t<To>, Tensor>::value;
			static_assert(IS_TENSOR_INSTANTIATION, "");
		}

	using CommonType = core::CSRSpMVOut<Kp, T1>;
	using TensorOut  = std::conditional_t<IS_OUT_DEFINED, To, Tensor<CommonType, B1>>;

	constexpr bool IS_SAME_DEVICE = (B1 == B2 && B1 == B3 && B1 == B4 && B1 == (std::remove_reference_t<TensorOut>::backend_type));
	static_assert(IS_SAME_DEVICE, "Error: Binary ops cannot be performed on tensors with different Backends!");

	const auto assign_out = [&]() -> TensorOut
		{
			if constexpr(IS_OUT_DEFINED)
				{
					return maybe_out;
				}
			else
				{
					return TensorOut(head.dims()[0] - 1);
				}
		};

	TensorOut out = assign_out();

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

	if constexpr(IS_OUT_DEFINED)
		{
			return;
		}
	else
		{
			return out;
		}
}

template <typename T1, Backend B1, typename T2, Backend B2, typename To = core::no_value_t>
auto
matmul(
	const Tensor<T1, B1>& lhs,
	const Tensor<T2, B2>& rhs,
	To&&                  maybe_out = {})
{
	constexpr bool IS_OUT_DEFINED = !std::is_same_v<To, core::no_value_t>;

	if constexpr(IS_OUT_DEFINED)
		{
			constexpr bool IS_MUTABLE = !std::is_const_v<std::remove_reference_t<To>>;
			static_assert(IS_MUTABLE, "");

			constexpr bool IS_TENSOR_INSTANTIATION = core::is_tensor_instantiation<std::remove_cvref_t<To>, Tensor>::value;
			static_assert(IS_TENSOR_INSTANTIATION, "");
		}

	using CommonType = std::common_type_t<T1, T2>;
	using TensorOut  = std::conditional_t<IS_OUT_DEFINED, To, Tensor<CommonType, B1>>;

	constexpr bool IS_SAME_DEVICE = (B1 == B2 && B1 == (std::remove_reference_t<TensorOut>::backend_type));
	static_assert(IS_SAME_DEVICE, "Error: Binary ops cannot be performed on tensors with different Backends!");

	const std::uint32_t lhs_shared = lhs.dims().back();
	const std::uint32_t rhs_shared = rhs.dims().front();

	if(lhs_shared != rhs_shared)
		{
			throw std::runtime_error("Matrix multiplication requires the left tensor's last dimension to equal the right tensor's first dimension.");
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

	for(auto it = std::next(rhs.dims().begin()), end = rhs.dims().end(); it != end; ++it)
		{
			const std::uint64_t dim = *it;

			rhs_combined *= dim;
			remainder.emplace_back(dim);
		}

	if(remainder.empty())
		{
			remainder.emplace_back(1);
		}

	const auto assign_out = [&]() -> TensorOut
		{
			if constexpr(IS_OUT_DEFINED)
				{
					return maybe_out;
				}
			else
				{
					return TensorOut(remainder);
				}
		};

	TensorOut out = assign_out();

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

	if constexpr(IS_OUT_DEFINED)
		{
			return;
		}
	else
		{
			return out;
		}
}

}

#endif
