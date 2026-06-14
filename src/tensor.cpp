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
	const std::uint64_t requested_size = __shape.length() * sizeof(float);

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

	const std::uint64_t requested_size = __shape.length() * sizeof(float);

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
	if(__ptr != nullptr)
		{
			cpu::Allocator::deallocate(__ptr);
		}
}

Tensor&
Tensor::operator=(
	const Tensor& other)
{
	if(this != &other)
		{
			if(__ptr != nullptr)
				{
					cpu::Allocator::deallocate(__ptr);
				}

			__shape  = other.__shape;
			__device = other.__device;

			const std::uint64_t requested_size = __shape.length() * sizeof(float);

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
			if(__ptr != nullptr)
				{
					cpu::Allocator::deallocate(__ptr);
				}

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

	Tensor dst(lhs.__shape, lhs.__device);

	if(lhs.__shape == rhs.__shape)
		{
			if(lhs.__device.is_cpu())
				{
					cpu::Operations::sum(dst.__ptr, lhs.__ptr, rhs.__ptr, lhs.__shape.length());
				}
		}
	else
		{
			try
				{
					const Shape broadcast = Shape::broadcast(lhs.__shape, rhs.__shape);

					if(lhs.__device.is_cpu())
						{
							cpu::Operations::sum(dst.__ptr, lhs.__ptr, rhs.__ptr, lhs.__shape.length(), lhs.__shape.dims(), broadcast.strides());
						}
				}
			catch(...)
				{
					throw std::runtime_error("Cannot add tensors with different shapes.");
				}
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

	Tensor dst(lhs.__shape, lhs.__device);

	if(lhs.__shape == rhs.__shape)
		{
			if(lhs.__device.is_cpu())
				{
					cpu::Operations::diff(dst.__ptr, lhs.__ptr, rhs.__ptr, lhs.__shape.length());
				}
		}
	else
		{
			try
				{
					const Shape broadcast = Shape::broadcast(lhs.__shape, rhs.__shape);

					if(lhs.__device.is_cpu())
						{
							cpu::Operations::diff(dst.__ptr, lhs.__ptr, rhs.__ptr, lhs.__shape.length(), lhs.__shape.dims(), broadcast.strides());
						}
				}
			catch(...)
				{
					throw std::runtime_error("Cannot subtract tensors with different shapes.");
				}
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

	Tensor dst(lhs.__shape, lhs.__device);

	if(lhs.__shape == rhs.__shape)
		{
			if(lhs.__device.is_cpu())
				{
					cpu::Operations::mul(dst.__ptr, lhs.__ptr, rhs.__ptr, lhs.__shape.length());
				}
		}
	else
		{
			try
				{
					const Shape broadcast = Shape::broadcast(lhs.__shape, rhs.__shape);

					if(lhs.__device.is_cpu())
						{
							cpu::Operations::mul(dst.__ptr, lhs.__ptr, rhs.__ptr, lhs.__shape.length(), lhs.__shape.dims(), broadcast.strides());
						}
				}
			catch(...)
				{
					throw std::runtime_error("Cannot diff tensors with different shapes.");
				}
		}

	return dst;
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

			if(__shape == other.__shape)
				{
					if(__device.is_cpu())
						{
							cpu::Operations::sum(__ptr, __ptr, other.__ptr, __shape.length());
						}
				}
			else
				{
					try
						{
							const Shape broadcast = Shape::broadcast(__shape, other.__shape);

							if(__device.is_cpu())
								{
									cpu::Operations::sum(__ptr, __ptr, other.__ptr, __shape.length(), __shape.dims(), broadcast.strides());
								}
						}
					catch(...)
						{
							throw std::runtime_error("Cannot add tensors with different shapes.");
						}
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

			if(__shape == other.__shape)
				{
					if(__device.is_cpu())
						{
							cpu::Operations::diff(__ptr, __ptr, other.__ptr, __shape.length());
						}
				}
			else
				{
					try
						{
							const Shape broadcast = Shape::broadcast(__shape, other.__shape);

							if(__device.is_cpu())
								{
									cpu::Operations::diff(__ptr, __ptr, other.__ptr, __shape.length(), __shape.dims(), broadcast.strides());
								}
						}
					catch(...)
						{
							throw std::runtime_error("Cannot multiply tensors with different shapes.");
						}
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

			if(__shape == other.__shape)
				{
					if(__device.is_cpu())
						{
							cpu::Operations::mul(__ptr, __ptr, other.__ptr, __shape.length());
						}
				}
			else
				{
					try
						{
							const Shape broadcast = Shape::broadcast(__shape, other.__shape);

							if(__device.is_cpu())
								{
									cpu::Operations::mul(__ptr, __ptr, other.__ptr, __shape.length(), __shape.dims(), broadcast.strides());
								}
						}
					catch(...)
						{
							throw std::runtime_error("Cannot multiply tensors with different shapes.");
						}
				}
		}

	return *this;
}

Tensor
Tensor::matmul(
	const Tensor& lhs,
	const Tensor& rhs)
{
	if(lhs.__device != rhs.__device)
		{
			throw std::runtime_error("Cannot multiply tensors located on different devices.");
		}

	const std::uint64_t lhs_shared = lhs.__shape[-1];
	const std::uint64_t rhs_shared = rhs.__shape[0];

	if(lhs_shared != rhs_shared)
		{
			throw std::runtime_error("Cannot multiply tensors: left tensor's last dimension must match right tensor's first dimension.");
		}

	std::vector<std::uint64_t> remainder = {};

	std::uint64_t lhs_combined = 1;

	for(std::uint64_t i = 0, end = lhs.__shape.size() - 1; i < end; ++i)
		{
			const std::uint64_t dim = lhs.__shape[i];

			lhs_combined *= dim;
			remainder.emplace_back(dim);
		}

	std::uint64_t rhs_combined = 1;

	for(std::uint64_t i = 1, end = rhs.__shape.size(); i < end; ++i)
		{
			const std::uint64_t dim = rhs.__shape[i];

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

std::uint64_t
Tensor::calculate_index(
	const std::uint64_t dims_pack[],
	const std::uint64_t num_arguments) const
{
	if(num_arguments != __shape.size())
		{
			throw std::runtime_error("Incorrect number of tensor indices.");
		}

	const std::vector<std::uint64_t>& strides = __shape.strides();
	const std::vector<std::uint64_t>& dims    = __shape.dims();

	std::uint64_t index = 0;

	for(std::uint64_t i = 0; i < num_arguments; ++i)
		{
			if(dims_pack[i] >= dims[i])
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

}
