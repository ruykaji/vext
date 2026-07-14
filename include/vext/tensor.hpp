#ifndef __VEXT_TENSOR_HPP__
#define __VEXT_TENSOR_HPP__

#include "vext/core/type.hpp"
#include <algorithm>
#include <initializer_list>
#include <queue>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <vext/core/cpu/allocator.hpp>
#include <vext/core/cpu/operations/csr_scatter.hpp>
#include <vext/core/cpu/operations/csr_spmv.hpp>
#include <vext/core/cpu/operations/elementwise_binary.hpp>
#include <vext/core/cpu/operations/elementwise_logical.hpp>
#include <vext/core/cpu/operations/elementwise_unary.hpp>
#include <vext/core/cpu/operations/linear_algebra.hpp>
#include <vext/core/cpu/operations/memory.hpp>
#include <vext/core/cpu/operations/reduction.hpp>

#if VEXT_CUDA
#include <vext/core/cuda/allocator.cuh>
#include <vext/core/cuda/operations/csr_scatter.cuh>
#include <vext/core/cuda/operations/csr_spmv.cuh>
#include <vext/core/cuda/operations/elementwise_binary.cuh>
#include <vext/core/cuda/operations/elementwise_logical.cuh>
#include <vext/core/cuda/operations/elementwise_unary.cuh>
#include <vext/core/cuda/operations/linear_algebra.cuh>
#include <vext/core/cuda/operations/memory.cuh>
#include <vext/core/cuda/operations/reduction.cuh>
#endif

#include <vext/type.hpp>

namespace vext
{

template <typename T1, Backend B1 = Backend::CPU>
class Tensor
{
	template <typename T2, Backend B2>
	friend class Tensor;

	struct InitializerDimension
	{
		T1                                value    = 0;
		std::vector<InitializerDimension> children = {};

		InitializerDimension(
			const T1 value)
			: value(value) {};

		InitializerDimension(
			const std::initializer_list<InitializerDimension>& children)
			: children(children) {};
	};

public:
	explicit Tensor(
		const std::vector<std::uint32_t>& dims) : __dims(dims)
	{
		compute_shape();
		allocate();
	}

	explicit Tensor(
		const std::initializer_list<InitializerDimension>& list)
	{
		if(list.size() == 0)
			{
				throw std::runtime_error("");
			}

		const InitializerDimension root(list);

		std::vector<std::uint32_t> dims;
		std::vector<T1>            data;

		for(InitializerDimension const* node = &root; !node->children.empty(); node = &node->children[0])
			{
				const std::uint64_t size = node->children.size();

				if(size == 1)
					{
						continue;
					}

				dims.emplace_back(size);
			}

		std::queue<const InitializerDimension*> queue;
		queue.emplace(&root);

		while(!queue.empty())
			{
				const InitializerDimension* node = queue.front();
				queue.pop();

				const std::vector<InitializerDimension>& children = node->children;

				if(children.empty())
					{
						throw std::invalid_argument("Zero-sized dimensions are not allowed.");
					}

				const std::uint64_t expected_size = children.front().children.size();

				for(const auto& child : children)
					{
						if(child.children.size() != expected_size)
							{
								throw std::invalid_argument("Inconsistent shape.");
							}

						if(child.children.empty())
							{
								data.emplace_back(child.value);
							}
						else
							{
								queue.emplace(&child);
							}
					}
			}

		__dims = std::move(dims);

		compute_shape();
		allocate<Backend::CPU>(data.data());
	}

	template <std::integral... Is>
	Tensor(
		Is... dims)
	{
		static_assert(sizeof...(dims) >= core::MIN_RANK, "");
		static_assert(sizeof...(dims) <= core::MAX_RANK, "");

		__dims = { static_cast<std::uint32_t>(dims)... };

		compute_shape();
		allocate();
	}

	template <Backend B2>
	Tensor(
		const Tensor<T1, B2>& other)
	{
		__length  = other.__length;
		__dims    = other.__dims;
		__strides = other.__strides;

		allocate<B2>(other.__ptr);
	}

	Tensor(
		const Tensor& other)
	{
		__length  = other.__length;
		__dims    = other.__dims;
		__strides = other.__strides;

		allocate<B1>(other.__ptr);
	}

	Tensor(
		Tensor&& other)
	{
		__length  = std::exchange(other.__length, core::MIN_LENGTH);
		__dims    = std::exchange(other.__dims, { core::MIN_LENGTH });
		__strides = std::exchange(other.__strides, {});
		__ptr     = std::exchange(other.__ptr, nullptr);
	}

	~Tensor()
	{
		deallocate();
	}

public:
	template <Backend B2>
	Tensor&
	operator=(
		const Tensor<T1, B2>& other)
	{
		if(this != &other)
			{
				deallocate();

				__length  = other.__length;
				__dims    = other.__dims;
				__strides = other.__strides;

				allocate<B2>(other.__ptr);
			}

		return *this;
	}

	Tensor&
	operator=(
		const Tensor& other)
	{
		if(this != &other)
			{
				deallocate();

				__length  = other.__length;
				__dims    = other.__dims;
				__strides = other.__strides;

				allocate<B1>(other.__ptr);
			}

		return *this;
	}

	Tensor&
	operator=(
		Tensor&& other)
	{
		if(this != &other)
			{
				deallocate();

				__length  = std::exchange(other.__length, core::MIN_LENGTH);
				__dims    = std::exchange(other.__dims, { core::MIN_LENGTH });
				__strides = std::exchange(other.__strides, {});

				__ptr = std::exchange(other.__ptr, nullptr);
			}

		return *this;
	}

	Tensor&
	operator=(
		T1 value)
	{
		if constexpr(B1 == Backend::CPU)
			{
				__ptr[0] = value;
			}
		#if VEXT_CUDA
		else
			{
				core::cuda::operations::memset(__ptr, value, 1);
			}
		#else
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
		#endif
	}

public:
	template <typename T2, Backend B2>
	friend auto
	operator==(
		const Tensor<T1, B1>& lhs,
		const Tensor<T2, B2>& rhs)
	{
		Tensor<std::uint8_t, B1> out(lhs.__dims);
		execute_logical_operation<core::LogicOperation::EQUAL>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	friend auto
	operator!=(
		const Tensor<T1, B1>& lhs,
		const Tensor<T2, B2>& rhs)
	{
		Tensor<std::uint8_t, B1> out(lhs.__dims);
		execute_logical_operation<core::LogicOperation::NOT_EQUAL>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	friend auto
	operator<(
		const Tensor<T1, B1>& lhs,
		const Tensor<T2, B2>& rhs)
	{
		Tensor<std::uint8_t, B1> out(lhs.__dims);
		execute_logical_operation<core::LogicOperation::LESS>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	friend auto
	operator<=(
		const Tensor<T1, B1>& lhs,
		const Tensor<T2, B2>& rhs)
	{
		Tensor<std::uint8_t, B1> out(lhs.__dims);
		execute_logical_operation<core::LogicOperation::LESS_EQUAL>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	friend auto
	operator>(
		const Tensor<T1, B1>& lhs,
		const Tensor<T2, B2>& rhs)
	{
		Tensor<std::uint8_t, B1> out(lhs.__dims);
		execute_logical_operation<core::LogicOperation::GREATER>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	friend auto
	operator>=(
		const Tensor<T1, B1>& lhs,
		const Tensor<T2, B2>& rhs)
	{
		Tensor<std::uint8_t, B1> out(lhs.__dims);
		execute_logical_operation<core::LogicOperation::GREATER_EQUAL>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	friend auto
	operator+(
		const Tensor<T1, B1>& lhs,
		const Tensor<T2, B2>& rhs)
	{
		Tensor<std::common_type_t<T1, T2>, B1> out(lhs.__dims);
		execute_binary_operation<core::BinaryOperation::ADD>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	friend auto
	operator-(
		const Tensor<T1, B1>& lhs,
		const Tensor<T2, B2>& rhs)
	{
		Tensor<std::common_type_t<T1, T2>, B1> out(lhs.__dims);
		execute_binary_operation<core::BinaryOperation::SUB>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	friend auto
	operator*(const Tensor<T1, B1>& lhs, const Tensor<T2, B2>& rhs)
	{
		Tensor<std::common_type_t<T1, T2>, B1> out(lhs.__dims);
		execute_binary_operation<core::BinaryOperation::MUL>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	friend auto
	operator/(
		const Tensor<T1, B1>& lhs,
		const Tensor<T2, B2>& rhs)
	{
		Tensor<std::common_type_t<T1, T2>, B1> out(lhs.__dims);
		execute_binary_operation<core::BinaryOperation::DIV>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	friend auto
	operator^(
		const Tensor<T1, B1>& lhs,
		const Tensor<T2, B2>& rhs)
	{
		Tensor<std::common_type_t<T1, T2>, B1> out(lhs.__dims);
		execute_binary_operation<core::BinaryOperation::POW>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	Tensor&
	operator+=(
		const Tensor<T2, B2>& rhs)
	{
		execute_binary_operation<core::BinaryOperation::ADD>(*this, *this, rhs);
		return *this;
	}

	template <typename T2, Backend B2>
	Tensor&
	operator-=(
		const Tensor<T2, B2>& rhs)
	{
		execute_binary_operation<core::BinaryOperation::SUB>(*this, *this, rhs);
		return *this;
	}

	template <typename T2, Backend B2>
	Tensor&
	operator*=(
		const Tensor<T2, B2>& rhs)
	{
		execute_binary_operation<core::BinaryOperation::MUL>(*this, *this, rhs);
		return *this;
	}

	template <typename T2, Backend B2>
	Tensor&
	operator/=(
		const Tensor<T2, B2>& rhs)
	{
		execute_binary_operation<core::BinaryOperation::DIV>(*this, *this, rhs);
		return *this;
	}

	template <typename T2, Backend B2>
	Tensor&
	operator^=(
		const Tensor<T2, B2>& rhs)
	{
		execute_binary_operation<core::BinaryOperation::POW>(*this, *this, rhs);
		return *this;
	}

	Tensor&
	operator-()
	{
		execute_unary_operation<core::UnaryOperation::NEG>(*this);
		return *this;
	}

public:
	void
	set_from(
		const std::vector<T1>& values)
	{
		if constexpr(B1 == Backend::CPU)
			{
				core::cpu::operations::memcpy<T1>(__ptr, values.data(), std::min<std::uint32_t>(values.size(), __length));
			}
		#if VEXT_CUDA
		else
			{
				core::cuda::operations::memcpy<T1, B1, Backend::CPU>(__ptr, values.data(), std::min<std::uint32_t>(values.size(), __length));
			}
		#else
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
		 #endif
	}

	template <typename T2, Backend B2>
	auto
	matmul(
		const Tensor<T2, B2>& rhs) const
	{
		static_assert(B1 == B2, "Error: Binary operations cannot be performed on tensors with different Backends!");

		const std::uint32_t lhs_shared = __dims.back();
		const std::uint32_t rhs_shared = rhs.__dims.front();

		if(lhs_shared != rhs_shared)
			{
				throw std::runtime_error("Cannot multiply tensors: left tensor's last dimension must match right tensor's first dimension.");
			}

		std::uint32_t lhs_combined = 1;
		std::uint32_t rhs_combined = 1;

		std::vector<std::uint32_t> remainder;
		remainder.reserve(__dims.size() + rhs.__dims.size());

		// clang-format off
        std::ranges::for_each(__dims.begin(), std::prev(__dims.end()), [&lhs_combined, &remainder](const std::uint32_t dim){ lhs_combined *= dim; remainder.emplace_back(dim); });
        std::ranges::for_each(std::next(rhs.__dims.begin()), rhs.__dims.end(), [&rhs_combined, &remainder](const std::uint32_t dim){ rhs_combined *= dim; remainder.emplace_back(dim); });
		// clang-format on

		if(remainder.empty())
			{
				remainder.emplace_back(1);
			}

		Tensor<std::common_type_t<T1, T2>, B1> out( remainder);

		if constexpr(B1 == Backend::CPU)
			{
				core::cpu::operations::matmul<T1, T2>(out.__ptr, __ptr, rhs.__ptr, lhs_combined, lhs_shared, rhs_combined);
			}
		#if VEXT_CUDA
		else
			{
				core::cuda::operations::matmul<T1, T2>(out.__ptr, __ptr, rhs.__ptr, lhs_combined, lhs_shared, rhs_combined);
			}
		#else
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
		#endif

		return out;
	}

	template <typename T2, Backend B2>
	void
	prelu(
		const Tensor<T2, B2>& rhs)
	{
		execute_binary_operation<core::BinaryOperation::PRELU>(*this, *this, rhs);
	}

	void
	abs()
	{
		execute_unary_operation<core::UnaryOperation::ABS>(*this);
	}

	void
	sin()
	{
		execute_unary_operation<core::UnaryOperation::SIN>(*this);
	}

	void
	cos()
	{
		execute_unary_operation<core::UnaryOperation::COS>(*this);
	}

	void
	exp()
	{
		execute_unary_operation<core::UnaryOperation::EXP>(*this);
	}

	void
	log()
	{
		execute_unary_operation<core::UnaryOperation::LOG>(*this);
	}

	void
	sqrt()
	{
		execute_unary_operation<core::UnaryOperation::SQRT>(*this);
	}

	void
	square()
	{
		execute_unary_operation<core::UnaryOperation::SQUARE>(*this);
	}

	void
	round()
	{
		execute_unary_operation<core::UnaryOperation::ROUND>(*this);
	}

	void
	sigmoid()
	{
		execute_unary_operation<core::UnaryOperation::SIGMOID>(*this);
	}

	void
	soft_relu()
	{
		execute_unary_operation<core::UnaryOperation::SOFT_RELU>(*this);
	}

	void
	relu()
	{
		execute_unary_operation<core::UnaryOperation::RELU>(*this);
	}

	void
	softmax()
	{
		execute_unary_operation<core::UnaryOperation::SOFTMAX>(*this);
	}

	void
	softmin()
	{
		execute_unary_operation<core::UnaryOperation::SOFTMIN>(*this);
	}

	void
	log_softmax()
	{
		execute_unary_operation<core::UnaryOperation::LOGSOFTMAX>(*this);
	}

	void
	leaky_relu(
		const float a = 0.0f)
	{
		execute_unary_operation<core::UnaryOperation::LEAKY_RELU>(*this, a);
	}

	void
	elu(
		const float a = 0.0f)
	{
		execute_unary_operation<core::UnaryOperation::ELU>(*this, a);
	}

	void
	swish(
		const float a = 0.0f)
	{
		execute_unary_operation<core::UnaryOperation::SWISH>(*this, a);
	}

	void
	linear(
		const float a = 1.0f,
		const float b = 0.0f)
	{
		execute_unary_operation<core::UnaryOperation::LINEAR>(*this, a, b);
	}

	void
	clip(
		const float a = std::numeric_limits<float>::lowest(),
		const float b = std::numeric_limits<float>::infinity())
	{
		execute_unary_operation<core::UnaryOperation::CLIP>(*this, a, b);
	}

	void
	pow(
		const float a = 1.0f,
		const float b = 0.0f)
	{
		execute_unary_operation<core::UnaryOperation::POW>(*this, a, b);
	}

	template <std::integral... Is>
	Tensor
	sum(
		Is... axis) const
	{
		return execute_reduction<core::ReductionOperation::SUM, T1>(*this, axis...);
	}

	template <std::integral... Is>
	Tensor
	prod(
		Is... axis) const
	{
		return execute_reduction<core::ReductionOperation::PROD, T1>(*this, axis...);
	}

	template <std::integral... Is>
	Tensor
	min(
		Is... axis) const
	{
		return execute_reduction<core::ReductionOperation::MIN, T1>(*this, axis...);
	}

	template <std::integral... Is>
	Tensor
	max(
		Is... axis) const
	{
		return execute_reduction<core::ReductionOperation::MAX, T1>(*this, axis...);
	}

	template <std::integral... Is>
	Tensor<float, B1>
	mean(
		Is... axis) const
	{
		return execute_reduction<core::ReductionOperation::MEAN, float>(*this, axis...);
	}

	template <std::integral... Is>
	Tensor<float, B1>
	var(
		Is... axis) const
	{
		return execute_reduction<core::ReductionOperation::VAR, float>(*this, axis...);
	}

	template <std::integral... Is>
	Tensor<float, B1>
	std(
		Is... axis) const
	{
		return execute_reduction<core::ReductionOperation::STD, float>(*this, axis...);
	}

    template <std::integral... Is>
	Tensor<float, B1>
	l2_norm(
		Is... axis) const
	{
		return execute_reduction<core::ReductionOperation::L2_NORM, float>(*this, axis...);
	}

	template <Backend B2, Backend B3>
	Tensor<T1, B1>
	csr_scatter_sum(
		const Tensor<std::uint32_t, B2>& head,
		const Tensor<std::uint32_t, B3>& tail) const
	{
		return execute_csr_scatter<core::CSRScatterOperation::SUM, T1, B2, B3>(*this, head, tail);
	}

	template <Backend B2, Backend B3>
	Tensor<float, B1>
	csr_scatter_mean(
		const Tensor<std::uint32_t, B2>& head,
		const Tensor<std::uint32_t, B3>& tail) const
	{
		return execute_csr_scatter<core::CSRScatterOperation::MEAN, float, B2, B3>(*this, head, tail);
	}

	template <Backend B2, Backend B3>
	Tensor<T1, B1>
	csr_scatter_min(
		const Tensor<std::uint32_t, B2>& head,
		const Tensor<std::uint32_t, B3>& tail) const
	{
		return execute_csr_scatter<core::CSRScatterOperation::MIN, T1, B2, B3>(*this, head, tail);
	}

	template <Backend B2, Backend B3>
	Tensor<T1, B1>
	csr_scatter_max(
		const Tensor<std::uint32_t, B2>& head,
		const Tensor<std::uint32_t, B3>& tail) const
	{
		return execute_csr_scatter<core::CSRScatterOperation::MAX, T1, B2, B3>(*this, head, tail);
	}

	template <Backend B2, Backend B3>
	Tensor<T1, B1>
	csr_scatter_prod(
		const Tensor<std::uint32_t, B2>& head,
		const Tensor<std::uint32_t, B3>& tail) const
	{
		return execute_csr_scatter<core::CSRScatterOperation::PROD, T1, B2, B3>(*this, head, tail);
	}

	template <Backend B2, Backend B3>
	Tensor<float, B1>
	csr_scatter_var(
		const Tensor<std::uint32_t, B2>& head,
		const Tensor<std::uint32_t, B3>& tail) const
	{
		return execute_csr_scatter<core::CSRScatterOperation::VAR, float, B2, B3>(*this, head, tail);
	}

	template <Backend B2, Backend B3>
	Tensor<float, B1>
	csr_scatter_std(
		const Tensor<std::uint32_t, B2>& head,
		const Tensor<std::uint32_t, B3>& tail) const
	{
		return execute_csr_scatter<core::CSRScatterOperation::STD, float, B2, B3>(*this, head, tail);
	}

	template <typename T2, Backend B2, Backend B3, Backend B4>
	Tensor<T1, B1>
	csr_spmv_sum(
		const Tensor<std::uint32_t, B2>& head,
		const Tensor<std::uint32_t, B3>& tail,
		const Tensor<T2, B4>&            x) const
	{
		return execute_csr_spmv<core::CSRSpMVOperation::SUM, T1, T2, B2, B3, B4>(*this, head, tail, x);
	}

	template <typename T2, Backend B2, Backend B3, Backend B4>
	Tensor<float, B1>
	csr_spmv_mean(
		const Tensor<std::uint32_t, B2>& head,
		const Tensor<std::uint32_t, B3>& tail,
		const Tensor<T2, B4>&            x) const
	{
		return execute_csr_spmv<core::CSRSpMVOperation::MEAN, float, T2, B2, B3, B4>(*this, head, tail, x);
	}

	template <typename T2, Backend B2, Backend B3, Backend B4>
	Tensor<T1, B1>
	csr_spmv_min(
		const Tensor<std::uint32_t, B2>& head,
		const Tensor<std::uint32_t, B3>& tail,
		const Tensor<T2, B4>&            x) const
	{
		return execute_csr_spmv<core::CSRSpMVOperation::MIN, T1, T2, B2, B3, B4>(*this, head, tail, x);
	}

	template <typename T2, Backend B2, Backend B3, Backend B4>
	Tensor<T1, B1>
	csr_spmv_max(
		const Tensor<std::uint32_t, B2>& head,
		const Tensor<std::uint32_t, B3>& tail,
		const Tensor<T2, B4>&            x) const
	{
		return execute_csr_spmv<core::CSRSpMVOperation::MAX, T1, T2, B2, B3, B4>(*this, head, tail, x);
	}

	template <typename T2, Backend B2, Backend B3, Backend B4>
	Tensor<T1, B1>
	csr_spmv_prod(
		const Tensor<std::uint32_t, B2>& head,
		const Tensor<std::uint32_t, B3>& tail,
		const Tensor<T2, B4>&            x) const
	{
		return execute_csr_spmv<core::CSRSpMVOperation::PROD, T1, T2, B2, B3, B4>(*this, head, tail, x);
	}

	template <typename T2, Backend B2, Backend B3, Backend B4>
	Tensor<float, B1>
	csr_spmv_var(
		const Tensor<std::uint32_t, B2>& head,
		const Tensor<std::uint32_t, B3>& tail,
		const Tensor<T2, B4>&            x) const
	{
		return execute_csr_spmv<core::CSRSpMVOperation::VAR, float, T2, B2, B3, B4>(*this, head, tail, x);
	}

	template <typename T2, Backend B2, Backend B3, Backend B4>
	Tensor<float, B1>
	csr_spmv_std(
		const Tensor<std::uint32_t, B2>& head,
		const Tensor<std::uint32_t, B3>& tail,
		const Tensor<T2, B4>&            x) const
	{
		return execute_csr_spmv<core::CSRSpMVOperation::STD, float, T2, B2, B3, B4>(*this, head, tail, x);
	}

	template <std::integral... Is>
	T1
	item(
		Is... dims) const
	{
		std::uint32_t index = 0;

		if constexpr(sizeof...(dims) == 1)
			{
				index = static_cast<std::uint32_t>((dims, ...));
			}
		else if constexpr(sizeof...(dims) > 1)
			{
				index = flat_index(dims...);

				if(index >= __length)
					{
						throw std::invalid_argument("");
					}
			}

		if constexpr(B1 == Backend::CPU)
			{
				return __ptr[index];
			}
		#if VEXT_CUDA
		else
			{
				return core::cuda::operations::memget(__ptr, index);
			}
		#else
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
		#endif
	}

	template <std::integral... Is>
	void
	put(
		const T1 value,
		Is... dims)
	{
		std::uint32_t index = 0;

		if constexpr(sizeof...(dims) == 1)
			{
				index = static_cast<std::uint32_t>((dims, ...));
			}
		else if constexpr(sizeof...(dims) > 1)
			{
				index = flat_index(dims...);

				if(index >= __length)
					{
						throw std::invalid_argument("");
					}
			}

		if constexpr(B1 == Backend::CPU)
			{
				__ptr[index] = value;
			}
		#if VEXT_CUDA
		else
			{
				return core::cuda::operations::memset(__ptr + index, value, 1);
			}
		#else
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
		#endif
	}

	std::uint32_t
	length() const noexcept
	{
		return __length;
	}

	const std::vector<std::uint32_t>&
	shape() const noexcept
	{
		return __dims;
	}

private:
	template <core::BinaryOperation Kp, typename T2, Backend B2>
	static void
	execute_binary_operation(
		Tensor<std::common_type_t<T1, T2>, B1>& out,
		const Tensor<T1, B1>&                   lhs,
		const Tensor<T2, B2>&                   rhs)
	{
		static_assert(B1 == B2, "Error: Binary operations cannot be performed on tensors with different Backends!");

		if(lhs.__dims == rhs.__dims)
			{
				if constexpr(B1 == Backend::CPU)
					{
						core::cpu::operations::binary<Kp, T1, T2>(out.__ptr, lhs.__ptr, rhs.__ptr, lhs.__length);
					}
				#if VEXT_CUDA
				else
					{
						core::cuda::operations::binary<Kp, T1, T2>(out.__ptr, lhs.__ptr, rhs.__ptr, lhs.__length);
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
				try
					{
						const std::vector<std::uint32_t> broadcast_strides = rhs.__length == 1 ? lhs.__strides : broadcast(lhs.__strides, lhs.__dims, rhs.__dims);

						if constexpr(B1 == Backend::CPU)
							{
								core::cpu::operations::binary_with_broadcast<Kp, T1, T2>(out.__ptr, lhs.__ptr, rhs.__ptr, lhs.__length, lhs.__dims, broadcast_strides);
							}
						#if VEXT_CUDA
						else
							{
								core::cuda::operations::binary_with_broadcast<Kp, T1, T2>(out.__ptr, lhs.__ptr, rhs.__ptr, lhs.__length, lhs.__dims, broadcast_strides);
							}
						#else
						else
							{
								static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
							}
						#endif
					}
				catch(...)
					{
						throw std::runtime_error("Cannot perform binary operation on incompatible shapes.");
					}
			}
	}

	template <core::UnaryOperation Kp, std::floating_point... Is>
	static void
	execute_unary_operation(
		Tensor<T1, B1>& out,
		Is... param)
	{
		if constexpr(B1 == Backend::CPU)
			{
				core::cpu::operations::unary<Kp, T1>(out.__ptr, out.__length, param...);
			}
		#if VEXT_CUDA
		else
			{
				core::cuda::operations::unary<Kp, T1>(out.__ptr, out.__length, param...);
			}
		#else
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
		#endif
	}

	template <core::LogicOperation Kp, typename T2, Backend B2>
	static void
	execute_logical_operation(
		Tensor<std::uint8_t, B1>& out,
		const Tensor<T1, B1>&     lhs,
		const Tensor<T2, B2>&     rhs)
	{
		static_assert(B1 == B2, "Error: Binary operations cannot be performed on tensors with different Backends!");

		if(lhs.__dims != rhs.__dims)
			{
				throw std::runtime_error("Cannot perform logical operation on incompatible shapes");
			}

		if constexpr(B1 == Backend::CPU)
			{
				core::cpu::operations::logical<Kp, T1, T2>(out.__ptr, lhs.__ptr, rhs.__ptr, out.__length);
			}
		#if VEXT_CUDA
		else
			{
				core::cuda::operations::logical<Kp, T1, T2>(out.__ptr, lhs.__ptr, rhs.__ptr, out.__length);
			}
		#else
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
		#endif
	}

	template <core::ReductionOperation Kp, typename T2, std::integral... Is>
	static auto
	execute_reduction(
		const Tensor<T1, B1>& src,
		Is... axis)
	{
		const std::uint32_t reduce_axis[] = { static_cast<std::uint32_t>(axis)... };
		const std::uint32_t reduce_size   = sizeof...(axis);

		const std::uint32_t dims_count = src.__dims.size();

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

				reduce_dims    = src.__dims;
				reduce_strides = src.__strides;
				M              = src.__length;
			}
		else
			{
				for(std::uint32_t i = 0; i < dims_count; ++i)
					{
						if(is_reduce_axis[i])
							{
								reduce_dims.emplace_back(src.__dims[i]);
								reduce_strides.emplace_back(src.__strides[i]);
								M *= src.__dims[i];
							}
						else
							{
								keep_dims.emplace_back(src.__dims[i]);
								keep_strides.emplace_back(src.__strides[i]);
								N *= src.__dims[i];
							}
					}
			}

		Tensor<T2, B1> out(keep_dims);

		if constexpr(B1 == Backend::CPU)
			{
				core::cpu::operations::reduce<Kp>(out.__ptr, src.__ptr, N, M, keep_dims, keep_strides, reduce_dims, reduce_strides);
			}
		#if VEXT_CUDA
		else
			{
				core::cuda::operations::reduce<Kp>(out.__ptr, src.__ptr, N, M, keep_dims, keep_strides, reduce_dims, reduce_strides);
			}
		#else
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
		#endif

		return out;
	}

	template <core::CSRScatterOperation Kp, typename T2, Backend B2, Backend B3>
	static auto
	execute_csr_scatter(
		const Tensor<T1, B1>&            src,
		const Tensor<std::uint32_t, B2>& head,
		const Tensor<std::uint32_t, B3>& tail)
	{
		static_assert(B1 == B2 && B1 == B3, "Error: CSR Scatter operations cannot be performed on tensors with different Backends!");

		if(src.__dims[0] != (head.__dims[0] - 1))
			{
				throw std::runtime_error("Cannot perform CSR Scatter operation on incompatible shapes");
			}

		Tensor<T2, B1> out(src.__dims);

		if constexpr(B1 == Backend::CPU)
			{
				core::cpu::operations::csr_scatter<Kp, T1, T2>(out.__ptr, src.__ptr, head.__ptr, tail.__ptr, out.__dims[0], out.__strides[0]);
			}
		#if VEXT_CUDA
		else
			{
				core::cuda::operations::csr_scatter<Kp, T1, T2>(out.__ptr, src.__ptr, head.__ptr, tail.__ptr, out.__dims[0], out.__strides[0]);
			}
		#else
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
		#endif

		return out;
	}

	template <core::CSRSpMVOperation Kp, typename T2, typename T3, Backend B2, Backend B3, Backend B4>
	static auto
	execute_csr_spmv(
		const Tensor<T1, B1>&            A,
		const Tensor<std::uint32_t, B2>& head,
		const Tensor<std::uint32_t, B3>& tail,
		const Tensor<T3, B4>&            x)
	{
		static_assert(B1 == B2 && B1 == B3 && B1 == B4, "Error: CSR Scatter operations cannot be performed on tensors with different Backends!");

		Tensor<T2, B1> out(head.__dims[0] - 1);

		if constexpr(B1 == Backend::CPU)
			{
				core::cpu::operations::csr_spmv<Kp, T2, T1, T3>(out.__ptr, A.__ptr, head.__ptr, tail.__ptr, x.__ptr, out.__length);
			}
		#if VEXT_CUDA
		else
			{
				core::cuda::operations::csr_spmv<Kp, T2, T1, T3>(out.__ptr, A.__ptr, head.__ptr, tail.__ptr, x.__ptr, out.__length);
			}
		#else
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
		#endif

		return out;
	}

	static std::vector<std::uint32_t>
	broadcast(
		const std::vector<std::uint32_t>& source_strides,
		const std::vector<std::uint32_t>& source_dims,
		const std::vector<std::uint32_t>& target_dims)
	{
		const std::uint64_t source_size = source_dims.size();
		const std::uint64_t target_size = target_dims.size();

		if(target_size > source_size)
			{
				throw std::runtime_error("");
			}

		std::uint64_t offset_left = 0;
		std::uint64_t subset_size = 0;

		for(std::uint64_t i = 0; i < source_size; ++i)
			{
				if(source_dims[i] == target_dims[subset_size])
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
				throw std::runtime_error("Unable to broadcast shapes");
			}

		std::vector<std::uint32_t> strides = source_strides;

		for(std::uint64_t i = 0; i < offset_left; ++i)
			{
				strides[i] = 0;
			}

		for(std::uint64_t i = offset_left + target_size; i < source_size; ++i)
			{
				strides[i] = 0;
			}

		return strides;
	}

	void
	compute_shape()
	{
		const std::uint64_t size = __dims.size();

		if(size < core::MIN_RANK)
			{
				throw std::runtime_error("");
			}

		if(size > core::MAX_RANK)
			{
				throw std::runtime_error("");
			}

		__strides.resize(size, 0);

		std::uint32_t stride = 1;

		for(std::uint64_t i = size; i > 0; --i)
			{
				__length *= __dims[i - 1];

				if(__length > core::MAX_LENGTH)
					{
						throw std::overflow_error("Length calculation overflowed");
					}

				__strides[i - 1] = stride;
				stride *= __dims[i - 1];
			}

		if(__length < core::MIN_LENGTH)
			{
				throw std::overflow_error("Length calculation underflowed");
			}
	}

	void
	allocate()
	{
		if constexpr(B1 == Backend::CPU)
			{
				__ptr = core::cpu::allocator::allocate<T1>(__length);
			}
		#if VEXT_CUDA
		else
			{
				__ptr = core::cuda::allocator::allocate<T1>(__length);
			}
		#else
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
		 #endif
	}

	template <Backend B2>
	void
	allocate(
		T1* data)
	{
		if constexpr(B1 == Backend::CPU)
			{
				__ptr = core::cpu::allocator::allocate<T1>(__length);

				if constexpr(B2 == Backend::CPU)
					{
						core::cpu::operations::memcpy<T1>(__ptr, data, __length);
					}
				#if VEXT_CUDA
				else
					{
						core::cuda::operations::memcpy<T1, B1, B2>(__ptr, data, __length);
					}
				#else
				else
					{
						static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
					}
				#endif
			}
		#if VEXT_CUDA
		else
			{
				__ptr = core::cuda::allocator::allocate<T1>(__length);
				core::cuda::operations::memcpy<T1, B1, B2>(__ptr, data, __length);
			}
		#else
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
		#endif
	}

	void
	deallocate()
	{
		if constexpr(B1 == Backend::CPU)
			{
				core::cpu::allocator::deallocate(__ptr);
			}
		#if VEXT_CUDA
		else
			{
				core::cuda::allocator::deallocate(__ptr);
			}
		#else
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
		#endif
	}

	template <std::integral... Is>
	std::uint64_t
	flat_index(
		Is... dims) const
	{
		const std::uint32_t dims_pack[]   = { static_cast<std::uint32_t>(dims)... };
		const std::uint32_t num_arguments = sizeof...(dims);

		if(num_arguments != __dims.size())
			{
				throw std::runtime_error("Incorrect number of tensor dimensions.");
			}

		std::uint32_t index = 0;

		for(std::uint32_t i = 0; i < num_arguments; ++i)
			{
				if(dims_pack[i] >= __dims[i])
					{
						throw std::runtime_error("Tensor index out of range.");
					}

				index += dims_pack[i] * __strides[i];
			}

		if(index >= __length)
			{
				throw std::runtime_error("");
			}

		return index;
	}

private:
	T1*                        __ptr     = nullptr;
	std::uint32_t              __length  = 1;
	std::vector<std::uint32_t> __dims    = { 1 };
	std::vector<std::uint32_t> __strides = {};
};
}

#endif
