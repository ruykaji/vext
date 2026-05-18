#include <cstring>
#include <utility>

#include "cpu/allocator.hpp"
#include "cpu/operations.hpp"

#include <vext/tensor.hpp>

namespace vext
{

Tensor::Tensor(
	const Shape  shape,
	const Device device)
	: __shape(shape),
	  __device(device)
{
	const std::size_t requested_size = __shape.length() * sizeof(float);

	if(__device.is_cpu())
		{
			__ptr = static_cast<float*>(cpu::Allocator::allocate(requested_size));
		}
}

Tensor::Tensor(
	const Tensor& other)
{
	__shape  = other.__shape;
	__device = other.__device;

	const std::size_t requested_size = __shape.length() * sizeof(float);

	if(__device.is_cpu())
		{
			__ptr = static_cast<float*>(cpu::Allocator::allocate(requested_size));
			std::memcpy(__ptr, other.__ptr, requested_size);
		}
}

Tensor::Tensor(
	Tensor&& other)
{
	__shape  = std::exchange(other.__shape, {});
	__device = std::exchange(other.__device, {});
	__ptr    = std::exchange(other.__ptr, nullptr);
}

Tensor::~Tensor()
{
	cpu::Allocator::deallocate(__ptr);
}

Tensor&
Tensor::operator=(
	const Tensor& other)
{
	if(this != &other)
		{
			__shape  = other.__shape;
			__device = other.__device;

			const std::size_t requested_size = __shape.length() * sizeof(float);

			if(__device.is_cpu())
				{
					__ptr = static_cast<float*>(cpu::Allocator::allocate(requested_size));
					std::memcpy(__ptr, other.__ptr, requested_size);
				}
		}

	return *this;
}

Tensor&
Tensor::operator=(
	Tensor&& other)
{
	if(this != &other)
		{
			__shape  = std::exchange(other.__shape, {});
			__device = std::exchange(other.__device, {});
			__ptr    = std::exchange(other.__ptr, nullptr);
		}

	return *this;
}

Tensor
operator+(
	const Tensor& lhs,
	const Tensor& rhs)
{
	if(lhs.__device != rhs.__device)
		{
			throw std::runtime_error("Cannot add tensors located on different devices.");
		}

	if(lhs.__shape != rhs.__shape)
		{
			throw std::runtime_error("Cannot add tensors with different shapes.");
		}

	Tensor dst(lhs.__shape, lhs.__device);

	if(lhs.__device.is_cpu())
		{
			cpu::Operations::sum(dst.__ptr, lhs.__ptr, rhs.__ptr, lhs.__shape.length());
		}

	return dst;
}

Tensor
operator-(
	const Tensor& lhs,
	const Tensor& rhs)
{
	if(lhs.__device != rhs.__device)
		{
			throw std::runtime_error("Cannot subtract tensors located on different devices.");
		}

	if(lhs.__shape != rhs.__shape)
		{
			throw std::runtime_error("Cannot subtract tensors with different shapes.");
		}

	Tensor dst(lhs.__shape, lhs.__device);

	if(lhs.__device.is_cpu())
		{
			cpu::Operations::diff(dst.__ptr, lhs.__ptr, rhs.__ptr, lhs.__shape.length());
		}

	return dst;
}

Tensor
operator*(
	const Tensor& lhs,
	const Tensor& rhs)
{
	if(lhs.__device != rhs.__device)
		{
			throw std::runtime_error("Cannot multiply tensors located on different devices.");
		}

	const std::size_t lhs_shared = lhs.__shape[-1];
	const std::size_t rhs_shared = rhs.__shape[0];

	if(lhs_shared != rhs_shared)
		{
			throw std::runtime_error("Cannot multiply tensors: left tensor's last dimension must match right tensor's first dimension.");
		}

	std::vector<std::size_t> remainder = {};

	std::size_t lhs_combined = 1;

	for(std::size_t i = 0, end = lhs.__shape.size() - 1; i < end; ++i)
		{
			const std::size_t dim = lhs.__shape[i];

			lhs_combined *= dim;
			remainder.emplace_back(dim);
		}

	std::size_t rhs_combined = 1;

	for(std::size_t i = 1, end = rhs.__shape.size(); i < end; ++i)
		{
			const std::size_t dim = rhs.__shape[i];

			rhs_combined *= dim;
			remainder.emplace_back(dim);
		}

	Shape  shape(remainder);
	Tensor tensor(shape, lhs.__device);

	if(lhs.__device.is_cpu())
		{
			cpu::Operations::mul(tensor.__ptr, lhs.__ptr, rhs.__ptr, lhs_combined, lhs_shared, rhs_combined);
		}

	return tensor;
}

Tensor&
Tensor::operator+=(
	const Tensor& other)
{
	if(this != &other)
		{
			if(__device != other.__device)
				{
					throw std::runtime_error("Cannot add tensors located on a different device.");
				}

			if(__shape != other.__shape)
				{
					throw std::runtime_error("Cannot add tensors with a different shape.");
				}

			if(__device.is_cpu())
				{
					cpu::Operations::sum(__ptr, other.__ptr, __shape.length());
				}
		}

	return *this;
}

Tensor&
Tensor::operator-=(
	const Tensor& other)
{
	if(this != &other)
		{
			if(__device != other.__device)
				{
					throw std::runtime_error("Cannot subtract tensors located on a different device.");
				}

			if(__shape != other.__shape)
				{
					throw std::runtime_error("Cannot subtract tensors with a different shape.");
				}

			if(__device.is_cpu())
				{
					cpu::Operations::diff(__ptr, other.__ptr, __shape.length());
				}
		}

	return *this;
}

Tensor&
Tensor::operator*=(
	const Tensor& other)
{
	if(this != &other)
		{
			if(__device != other.__device)
				{
					throw std::runtime_error("Cannot multiply tensors located on a different device.");
				}

			if(__shape != other.__shape)
				{
					throw std::runtime_error("Cannot multiply tensors with a different shape.");
				}

			if(__device.is_cpu())
				{
					cpu::Operations::mul(__ptr, other.__ptr, __shape.length());
				}
		}

	return *this;
}

}
