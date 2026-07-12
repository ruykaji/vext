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

#include <vext/shape.hpp>
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
			const std::initializer_list<InitializerDimension> children)
			: children(children) {};
	};

public:
	Tensor(
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

		__shape = std::move(Shape(dims));
		allocate<Backend::CPU>(data.data());
	}

	template <std::integral... Is>
	Tensor(
		Is... dims)
		: __shape(dims...)
	{
		allocate();
	}

	Tensor(
		const Shape& shape) : __shape(shape)
	{
		allocate();
	}

	template <Backend B2>
	Tensor(
		const Tensor<T1, B2>& other)
	{
		__shape = other.__shape;
		allocate<B2>(other.__ptr);
	}

	Tensor(
		const Tensor& other)
	{
		__shape = other.__shape;
		allocate<B1>(other.__ptr);
	}

	Tensor(
		Tensor&& other)
	{
		__shape = std::exchange(other.__shape, { core::MIN_LENGTH });
		__ptr   = std::exchange(other.__ptr, nullptr);
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

				__shape = other.__shape;
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

				__shape = other.__shape;
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

				__shape = std::exchange(other.__shape, {});
				__ptr   = std::exchange(other.__ptr, nullptr);
			}

		return *this;
	}

public:
	template <typename T2, Backend B2>
	friend auto
	operator==(
		const Tensor<T1, B1>& lhs,
		const Tensor<T2, B2>& rhs)
	{
		Tensor<std::uint8_t, B1> out(lhs.__shape);
		execute_logical_operation<core::LogicOperation::EQUAL>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	friend auto
	operator!=(
		const Tensor<T1, B1>& lhs,
		const Tensor<T2, B2>& rhs)
	{
		Tensor<std::uint8_t, B1> out(lhs.__shape);
		execute_logical_operation<core::LogicOperation::NOT_EQUAL>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	friend auto
	operator<(
		const Tensor<T1, B1>& lhs,
		const Tensor<T2, B2>& rhs)
	{
		Tensor<std::uint8_t, B1> out(lhs.__shape);
		execute_logical_operation<core::LogicOperation::LESS>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	friend auto
	operator<=(
		const Tensor<T1, B1>& lhs,
		const Tensor<T2, B2>& rhs)
	{
		Tensor<std::uint8_t, B1> out(lhs.__shape);
		execute_logical_operation<core::LogicOperation::LESS_EQUAL>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	friend auto
	operator>(
		const Tensor<T1, B1>& lhs,
		const Tensor<T2, B2>& rhs)
	{
		Tensor<std::uint8_t, B1> out(lhs.__shape);
		execute_logical_operation<core::LogicOperation::GREATER>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	friend auto
	operator>=(
		const Tensor<T1, B1>& lhs,
		const Tensor<T2, B2>& rhs)
	{
		Tensor<std::uint8_t, B1> out(lhs.__shape);
		execute_logical_operation<core::LogicOperation::GREATER_EQUAL>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	friend auto
	operator+(
		const Tensor<T1, B1>& lhs,
		const Tensor<T2, B2>& rhs)
	{
		Tensor<std::common_type_t<T1, T2>, B1> out(lhs.__shape);
		execute_binary_operation<core::BinaryOperation::ADD>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	friend auto
	operator-(
		const Tensor<T1, B1>& lhs,
		const Tensor<T2, B2>& rhs)
	{
		Tensor<std::common_type_t<T1, T2>, B1> out(lhs.__shape);
		execute_binary_operation<core::BinaryOperation::SUB>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	friend auto
	operator*(const Tensor<T1, B1>& lhs, const Tensor<T2, B2>& rhs)
	{
		Tensor<std::common_type_t<T1, T2>, B1> out(lhs.__shape);
		execute_binary_operation<core::BinaryOperation::MUL>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	friend auto
	operator/(
		const Tensor<T1, B1>& lhs,
		const Tensor<T2, B2>& rhs)
	{
		Tensor<std::common_type_t<T1, T2>, B1> out(lhs.__shape);
		execute_binary_operation<core::BinaryOperation::DIV>(out, lhs, rhs);
		return out;
	}

	template <typename T2, Backend B2>
	friend auto
	operator^(
		const Tensor<T1, B1>& lhs,
		const Tensor<T2, B2>& rhs)
	{
		Tensor<std::common_type_t<T1, T2>, B1> out(lhs.__shape);
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
				core::cpu::operations::memcpy<T1>(__ptr, values.data(), std::min<std::uint32_t>(values.size(), __shape.length()));
			}
		#if VEXT_CUDA
		else if constexpr(B1 == Backend::CUDA)
			{
				core::cuda::operations::memcpy<T1, B1, Backend::CPU>(__ptr, values.data(), std::min<std::uint32_t>(values.size(), __shape.length()));
			}
		#endif
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
	}

	template <typename T2, Backend B2>
	auto
	matmul(
		const Tensor<T2, B2>& rhs) const
	{
		static_assert(B1 == B2, "Error: Binary operations cannot be performed on tensors with different Backends!");

		const std::uint32_t lhs_shared = __shape[-1];
		const std::uint32_t rhs_shared = rhs.__shape[0];

		if(lhs_shared != rhs_shared)
			{
				throw std::runtime_error("Cannot multiply tensors: left tensor's last dimension must match right tensor's first dimension.");
			}

		std::uint32_t lhs_combined = 1;
		std::uint32_t rhs_combined = 1;

		std::vector<std::uint32_t> remainder;
		remainder.reserve(__shape.size() + rhs.__shape.size());

		// clang-format off
        std::ranges::for_each(__shape.begin(), std::prev(__shape.end()), [&lhs_combined, &remainder](const std::uint32_t dim){ lhs_combined *= dim; remainder.emplace_back(dim); });
        std::ranges::for_each(std::next(rhs.__shape.begin()), rhs.__shape.end(), [&rhs_combined, &remainder](const std::uint32_t dim){ rhs_combined *= dim; remainder.emplace_back(dim); });
		// clang-format on

		const Shape                            shape(std::move(remainder));
		Tensor<std::common_type_t<T1, T2>, B1> out(shape);

		if constexpr(B1 == Backend::CPU)
			{
				core::cpu::operations::matmul<T1, T2>(out.__ptr, __ptr, rhs.__ptr, lhs_combined, lhs_shared, rhs_combined);
			}
		#if VEXT_CUDA
		else if constexpr(B1 == Backend::CUDA)
			{
				core::cuda::operations::matmul<T1, T2>(out.__ptr, __ptr, rhs.__ptr, lhs_combined, lhs_shared, rhs_combined);
			}
		#endif
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}

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

				if(index >= __shape.length())
					{
						throw std::invalid_argument("");
					}
			}

		if constexpr(B1 == Backend::CPU)
			{
				return __ptr[index];
			}
		#if VEXT_CUDA
		else if constexpr(B1 == Backend::CUDA)
			{
				return core::cuda::operations::memget(__ptr, index);
			}
		#endif
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
	}

	const Shape&
	shape() const noexcept
	{
		return __shape;
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

		if(lhs.__shape == rhs.__shape)
			{
				if constexpr(B1 == Backend::CPU)
					{
						core::cpu::operations::binary<Kp, T1, T2>(out.__ptr, lhs.__ptr, rhs.__ptr, lhs.__shape.length());
					}
				#if VEXT_CUDA
				else if constexpr(B1 == Backend::CUDA)
					{
						core::cuda::operations::binary<Kp, T1, T2>(out.__ptr, lhs.__ptr, rhs.__ptr, lhs.__shape.length());
					}
				#endif
				else
					{
						static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
					}
			}
		else
			{
				try
					{
						const Shape broadcast = Shape::broadcast(lhs.__shape, rhs.__shape);

						if constexpr(B1 == Backend::CPU)
							{
								core::cpu::operations::binary_with_broadcast<Kp, T1, T2>(out.__ptr, lhs.__ptr, rhs.__ptr, lhs.__shape.length(), lhs.__shape.dims(), broadcast.strides());
							}
						#if VEXT_CUDA
						else if constexpr(B1 == Backend::CUDA)
							{
								core::cuda::operations::binary_with_broadcast<Kp, T1, T2>(out.__ptr, lhs.__ptr, rhs.__ptr, lhs.__shape.length(), lhs.__shape.dims(), broadcast.strides());
							}
						#endif
						else
							{
								static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
							}
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
				core::cpu::operations::unary<Kp, T1>(out.__ptr, out.__shape.length(), param...);
			}
		#if VEXT_CUDA
		else if constexpr(B1 == Backend::CUDA)
			{
				core::cuda::operations::unary<Kp, T1>(out.__ptr, out.__shape.length(), param...);
			}
		#endif
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
	}

	template <core::LogicOperation Kp, typename T2, Backend B2>
	static void
	execute_logical_operation(
		Tensor<std::uint8_t, B1>& out,
		const Tensor<T1, B1>&     lhs,
		const Tensor<T2, B2>&     rhs)
	{
		static_assert(B1 == B2, "Error: Binary operations cannot be performed on tensors with different Backends!");

		if(lhs.__shape != rhs.__shape)
			{
				throw std::runtime_error("Cannot perform logical operation on incompatible shapes");
			}

		if constexpr(B1 == Backend::CPU)
			{
				core::cpu::operations::logical<Kp, T1, T2>(out.__ptr, lhs.__ptr, rhs.__ptr, out.__shape.length());
			}
		#if VEXT_CUDA
		else if constexpr(B1 == Backend::CUDA)
			{
				core::cuda::operations::logical<Kp, T1, T2>(out.__ptr, lhs.__ptr, rhs.__ptr, out.__shape.length());
			}
		#endif
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
	}

	template <core::ReductionOperation Kp, typename T2, std::integral... Is>
	static auto
	execute_reduction(
		const Tensor<T1, B1>& src,
		Is... axis)
	{
		const std::uint32_t reduce_axis[] = { static_cast<std::uint32_t>(axis)... };
		const std::uint32_t reduce_size   = sizeof...(axis);

		const std::vector<std::uint32_t>& strides = src.__shape.strides();
		const std::vector<std::uint32_t>& dims    = src.__shape.dims();

		const std::uint32_t dims_count = dims.size();

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

				reduce_dims    = dims;
				reduce_strides = strides;
				M              = src.__shape.length();
			}
		else
			{
				for(std::uint32_t i = 0; i < dims_count; ++i)
					{
						if(is_reduce_axis[i])
							{
								reduce_dims.emplace_back(dims[i]);
								reduce_strides.emplace_back(strides[i]);
								M *= dims[i];
							}
						else
							{
								keep_dims.emplace_back(dims[i]);
								keep_strides.emplace_back(strides[i]);
								N *= dims[i];
							}
					}
			}

		const Shape    shape(keep_dims);
		Tensor<T2, B1> out(shape);

		if constexpr(B1 == Backend::CPU)
			{
				core::cpu::operations::reduce<Kp>(out.__ptr, src.__ptr, N, M, keep_dims, keep_strides, reduce_dims, reduce_strides);
			}
		#if VEXT_CUDA
		else if constexpr(B1 == Backend::CUDA)
			{
				core::cuda::operations::reduce<Kp>(out.__ptr, src.__ptr, N, M, keep_dims, keep_strides, reduce_dims, reduce_strides);
			}
		 #endif
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}

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

		if(src.__shape[0] != (head.__shape[0] - 1))
			{
				throw std::runtime_error("Cannot perform CSR Scatter operation on incompatible shapes");
			}

		Tensor<T2, B1> out(src.__shape);

		if constexpr(B1 == Backend::CPU)
			{
				core::cpu::operations::csr_scatter<Kp, T1, T2>(out.__ptr, src.__ptr, head.__ptr, tail.__ptr, out.__shape[0], out.__shape.strides()[0]);
			}
		#if VEXT_CUDA
		else if constexpr(B1 == Backend::CUDA)
			{
				core::cuda::operations::csr_scatter<Kp, T1, T2>(out.__ptr, src.__ptr, head.__ptr, tail.__ptr, out.__shape[0], out.__shape.strides()[0]);
			}
		#endif
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}

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

		Tensor<T2, B1> out(head.__shape[0] - 1);

		if constexpr(B1 == Backend::CPU)
			{
				core::cpu::operations::csr_spmv<Kp, T2, T1, T3>(out.__ptr, A.__ptr, head.__ptr, tail.__ptr, x.__ptr, out.__shape.length());
			}
		#if VEXT_CUDA
		else if constexpr(B1 == Backend::CUDA)
			{
				core::cuda::operations::csr_spmv<Kp, T2, T1, T3>(out.__ptr, A.__ptr, head.__ptr, tail.__ptr, x.__ptr, out.__shape.length());
			}
		#endif
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}

		return out;
	}

	void
	allocate()
	{
		const std::uint64_t count = __shape.length();

		if constexpr(B1 == Backend::CPU)
			{
				__ptr = core::cpu::allocator::allocate<T1>(count);
			}
		#if VEXT_CUDA
		else if constexpr(B1 == Backend::CUDA)
			{
				__ptr = core::cuda::allocator::allocate<T1>(count);
			}
		#endif
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
	}

	template <Backend B2>
	void
	allocate(
		T1* data)
	{
		const std::uint64_t count = __shape.length();

		if constexpr(B1 == Backend::CPU)
			{
				__ptr = core::cpu::allocator::allocate<T1>(count);

				if constexpr(B2 == Backend::CPU)
					{
						core::cpu::operations::memcpy<T1>(__ptr, data, count);
					}
				#if VEXT_CUDA
				else if constexpr(B2 == Backend::CUDA)
					{
						core::cuda::operations::memcpy<T1, B1, B2>(__ptr, data, count);
					}
				#endif
				else
					{
						static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
					}
			}
		#if VEXT_CUDA
		else if constexpr(B1 == Backend::CUDA)
			{
				__ptr = core::cuda::allocator::allocate<T1>(count);

				core::cuda::operations::memcpy<T1, B1, B2>(__ptr, data, count);
			}
		#endif
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
	}

	void
	deallocate()
	{
		if constexpr(B1 == Backend::CPU)
			{
				core::cpu::allocator::deallocate(__ptr);
			}
		#if VEXT_CUDA
		else if constexpr(B1 == Backend::CUDA)
			{
				core::cuda::allocator::deallocate(__ptr);
			}
		#endif
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
	}

	template <std::integral... Is>
	std::uint64_t
	flat_index(
		Is... dims) const
	{
		const std::uint32_t dims_pack[]   = { static_cast<std::uint32_t>(dims)... };
		const std::uint32_t num_arguments = sizeof...(dims);

		if(num_arguments != __shape.size())
			{
				throw std::runtime_error("Incorrect number of tensor dimensions.");
			}

		const std::vector<std::uint32_t>& strides = __shape.strides();

		std::uint32_t index = 0;

		for(std::uint32_t i = 0; i < num_arguments; ++i)
			{
				if(dims_pack[i] >= __shape[i])
					{
						throw std::runtime_error("Tensor index out of range.");
					}

				index += dims_pack[i] * strides[i];
			}

		const std::uint32_t length = __shape.length();

		if(index >= length)
			{
				throw std::runtime_error("");
			}

		return index;
	}

private:
	T1*   __ptr   = nullptr;
	Shape __shape = { core::MIN_LENGTH };
};

}

#endif
