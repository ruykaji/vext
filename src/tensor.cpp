#include <cstring>

#include "cpu/allocator.hpp"
#include <utility>
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
			__ptr = AllocatorCPU::allocate(requested_size);
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
			__ptr = AllocatorCPU::allocate(requested_size);
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
	AllocatorCPU::deallocate(__ptr);
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
					__ptr = AllocatorCPU::allocate(requested_size);
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
}

Tensor
operator-(
	const Tensor& lhs,
	const Tensor& rhs)
{
}

Tensor
operator*(
	const Tensor& lhs,
	const Tensor& rhs)
{
}

}
