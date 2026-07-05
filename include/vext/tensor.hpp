#ifndef __VEXT_TENSOR_HPP__
#define __VEXT_TENSOR_HPP__

#include <algorithm>
#include <initializer_list>
#include <queue>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <vext/core/cpu/allocator.hpp>
#include <vext/core/cpu/operations/elementwise_binary.hpp>
#include <vext/core/cpu/operations/elementwise_logical.hpp>
#include <vext/core/cpu/operations/elementwise_unary.hpp>
#include <vext/core/cpu/operations/linear_algebra.hpp>
#include <vext/core/cpu/operations/reduction.hpp>
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
		const InitializerDimension root(list);

		std::vector<std::uint64_t> dims;
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

		__shape = Shape(dims);

		allocate(data.data());
	}

	template <std::integral... Is>
	Tensor(
		Is... dims) : __shape(dims...)
	{
		allocate();
	}

	Tensor(
		const Shape& shape = {}) : __shape(shape)
	{
		allocate();
	}

	template <typename T2, Backend B2>
	Tensor(
		const Tensor<T2, B2>& other)
	{
		__shape = other.__shape;

		if constexpr(B1 == B2)
			{
				allocate(other.__ptr);
			}
		else
			{
			}
	}

	template <Backend B2>
	Tensor(
		const Tensor<T1, B2>& other)
	{
		if constexpr(B1 == Backend::CPU)
			{
			}
		else
			{
			}
	}

	Tensor(
		const Tensor& other)
	{
		__shape = other.__shape;

		allocate(other.__ptr);
	}

	Tensor(
		Tensor&& other)
	{
		__shape = std::exchange(other.__shape, {});
		__ptr   = std::exchange(other.__ptr, nullptr);
	}

	~Tensor()
	{
		if constexpr(B1 == Backend::CPU)
			{
				core::cpu::deallocate(__ptr);
			}
	}

public:
	template <typename T2, Backend B2>
	Tensor&
	operator=(
		const Tensor<T2, B2>& other)
	{
		if(this != &other)
			{
				if(__ptr != nullptr)
					{
						core::cpu::deallocate(__ptr);
					}

				__shape = other.__shape;

				if constexpr(B1 == B2)
					{
						allocate(other.__ptr);
					}
				else
					{
					}
			}

		return *this;
	}

	Tensor&
	operator=(
		const Tensor& other)
	{
		if(this != &other)
			{
				if(__ptr != nullptr)
					{
						core::cpu::deallocate(__ptr);
					}

				__shape = other.__shape;

				allocate(other.__ptr);
			}

		return *this;
	}

	Tensor&
	operator=(
		Tensor&& other)
	{
		if(this != &other)
			{
				if(__ptr != nullptr)
					{
						core::cpu::deallocate(__ptr);
					}

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
	template <typename T2, Backend B2>
	auto
	matmul(
		const Tensor<T2, B2>& rhs) const
	{
		static_assert(B1 == B2, "Error: Binary operations cannot be performed on tensors with different Backends!");

		const std::uint64_t lhs_shared = __shape[-1];
		const std::uint64_t rhs_shared = rhs.__shape[0];

		if(lhs_shared != rhs_shared)
			{
				throw std::runtime_error("Cannot multiply tensors: left tensor's last dimension must match right tensor's first dimension.");
			}

		std::uint64_t lhs_combined = 1;
		std::uint64_t rhs_combined = 1;

		std::vector<std::uint64_t> remainder;
		remainder.reserve(__shape.size() + rhs.__shape.size());

		// clang-format off
        std::ranges::for_each(__shape.begin(), std::prev(__shape.end()), [&lhs_combined, &remainder](const std::uint64_t dim){ lhs_combined *= dim; remainder.emplace_back(dim); });
        std::ranges::for_each(std::next(rhs.__shape.begin()), rhs.__shape.end(), [&rhs_combined, &remainder](const std::uint64_t dim){ rhs_combined *= dim; remainder.emplace_back(dim); });
		// clang-format on

		const Shape                            shape(std::move(remainder));
		Tensor<std::common_type_t<T1, T2>, B1> out(shape);

		if constexpr(B1 == Backend::CPU)
			{
				core::cpu::operations::matmul<T1, T2>(out.__ptr, __ptr, rhs.__ptr, lhs_combined, lhs_shared, rhs_combined);
			}

		return out;
	}

	void
	abs()
	{
		core::cpu::operations::unary<core::UnaryOperation::ABS>(__ptr, __shape.length());
	}

	void
	sin()
	{
		core::cpu::operations::unary<core::UnaryOperation::SIN>(__ptr, __shape.length());
	}

	void
	cos()
	{
		core::cpu::operations::unary<core::UnaryOperation::COS>(__ptr, __shape.length());
	}

	void
	exp()
	{
		core::cpu::operations::unary<core::UnaryOperation::EXP>(__ptr, __shape.length());
	}

	void
	log()
	{
		core::cpu::operations::unary<core::UnaryOperation::LOG>(__ptr, __shape.length());
	}

	void
	sqrt()
	{
		core::cpu::operations::unary<core::UnaryOperation::SQRT>(__ptr, __shape.length());
	}

	void
	sigmoid()
	{
		core::cpu::operations::unary<core::UnaryOperation::SIGMOID>(__ptr, __shape.length());
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
	T1&
	item(
		Is... dims)
	{
		const std::uint64_t index = flat_index(dims...);

		if(index >= __shape.length())
			{
				throw std::invalid_argument("");
			}

		if constexpr(B1 == Backend::CPU)
			{
				return __ptr[index];
			}
		else
			{
			}
	}

	template <std::integral... Is>
	T1
	item(
		Is... dims) const
	{
		const std::uint64_t index = flat_index(dims...);

		if(index >= __shape.length())
			{
				throw std::invalid_argument("");
			}

		if constexpr(B1 == Backend::CPU)
			{
				return __ptr[index];
			}
		else
			{
			}
	}

	T1&
	flat_item(
		const std::uint64_t index)
	{
		if(index >= __shape.length())
			{
				throw std::invalid_argument("");
			}

		if constexpr(B1 == Backend::CPU)
			{
				return __ptr[index];
			}
		else
			{
			}
	}

	T1
	flat_item(
		const std::uint64_t index) const
	{
		if(index >= __shape.length())
			{
				throw std::invalid_argument("");
			}

		if constexpr(B1 == Backend::CPU)
			{
				return __ptr[index];
			}
		else
			{
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
				else
					{
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
						else
							{
							}
					}
				catch(...)
					{
						throw std::runtime_error("Cannot perform binary operation on incompatible shapes.");
					}
			}
	}

	template <core::UnaryOperation Kp>
	static void
	execute_unary_operation(
		Tensor<T1, B1>& out)
	{
		if constexpr(B1 == Backend::CPU)
			{
				core::cpu::operations::unary<Kp, T1>(out.__ptr, out.__shape.length());
			}
		else
			{
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
		else
			{
			}
	}

	template <core::ReductionOperation Kp, typename T2, std::integral... Is>
	static auto
	execute_reduction(
		const Tensor<T1, B1>& src,
		Is... axis)
	{
		const std::int64_t  reduce_axis[] = { static_cast<std::int64_t>(axis)... };
		const std::uint64_t reduce_size   = sizeof...(axis);

		const std::vector<std::uint64_t>& strides = src.__shape.strides();
		const std::vector<std::uint64_t>& dims    = src.__shape.dims();

		const std::uint64_t dims_count = dims.size();

		if(reduce_size > dims_count)
			{
				throw std::runtime_error("Cannot reduce tensor along non-existing axis.");
			}

		std::vector<std::uint8_t> is_reduce_axis(dims_count, 0);

		for(std::uint64_t i = 0; i < reduce_size; ++i)
			{
				std::int64_t ax = reduce_axis[i];

				if(ax < 0)
					{
						ax += dims_count;
					}

				if(ax >= dims_count || ax < 0)
					{
						throw std::runtime_error("Cannot reduce tensor along non-existing axis.");
					}

				if(is_reduce_axis[ax])
					{
						throw std::runtime_error("Duplicate reduction axis.");
					}

				is_reduce_axis[ax] = true;
			}

		const std::uint64_t keep_size = reduce_size == 0 ? 1 : (dims_count - reduce_size);

		std::vector<std::uint64_t> keep_dims;
		std::vector<std::uint64_t> keep_strides;
		std::vector<std::uint64_t> reduce_dims;
		std::vector<std::uint64_t> reduce_strides;

		keep_dims.reserve(keep_size);
		keep_strides.reserve(keep_size);
		reduce_dims.reserve(reduce_size);
		reduce_strides.reserve(reduce_size);

		std::uint64_t N = 1;
		std::uint64_t M = 1;

		if(reduce_size == 0)
			{
				keep_dims.emplace_back(1);
				keep_strides.emplace_back(0);

				reduce_dims    = dims;
				reduce_strides = strides;
				M              = src.__shape.length();
			}
		else
			{
				for(std::uint64_t i = 0; i < dims_count; ++i)
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
		else
			{
			}

		return out;
	}

	template <typename... Args>
	void
	allocate(
		Args... args)
	{
		const std::uint64_t requested_size = __shape.length() * sizeof(T1);

		if constexpr(B1 == Backend::CPU)
			{
				__ptr = static_cast<T1*>(core::cpu::allocate(requested_size));

				if constexpr(sizeof...(args) != 0)
					{
						std::memcpy(__ptr, static_cast<void*>(args)..., requested_size);
					}
			}
		else
			{
			}
	}

	template <std::integral... Is>
	std::uint64_t
	flat_index(
		Is... dims) const
	{
		const std::uint64_t dims_pack[]   = { static_cast<std::uint64_t>(dims)... };
		const std::uint64_t num_arguments = sizeof...(dims);

		if(num_arguments != __shape.size())
			{
				throw std::runtime_error("Incorrect number of tensor dimensions.");
			}

		const std::vector<std::uint64_t>& strides = __shape.strides();

		std::uint64_t index = 0;

		for(std::uint64_t i = 0; i < num_arguments; ++i)
			{
				if(dims_pack[i] >= __shape[i])
					{
						throw std::runtime_error("Tensor index out of range.");
					}

				index += dims_pack[i] * strides[i];
			}

		const std::uint64_t length = __shape.length();

		if(index >= length)
			{
				throw std::runtime_error("");
			}

		return index;
	}

private:
	T1*           __ptr         = nullptr;
	std::uint64_t __cpu_stream  = 0;
	std::uint64_t __cuda_stream = 0;
	Shape         __shape       = {};
};
}

#endif
