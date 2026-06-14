#ifndef __VEXT_TENSOR_HPP__
#define __VEXT_TENSOR_HPP__

#include <vext/device.hpp>
#include <vext/shape.hpp>
#include <vext/type.hpp>

namespace vext
{

class Tensor

{
public:
	Tensor(
		const Shape  shape  = {},
		const Device device = {});

	Tensor(
		const Tensor&);

	Tensor(
		Tensor&&);

	~Tensor();

public:
	Tensor&
	operator=(
		const Tensor&);

	Tensor&
	operator=(
		Tensor&&);

public:
	friend Tensor
	operator+(
		const Tensor& lhs,
		const Tensor& rhs);

	friend Tensor
	operator-(
		const Tensor& lhs,
		const Tensor& rhs);

	friend Tensor
	operator*(
		const Tensor& lhs,
		const Tensor& rhs);

	Tensor&
	operator+=(
		const Tensor& other);

	Tensor&
	operator-=(
		const Tensor& other);

	Tensor&
	operator*=(
		const Tensor& other);

	template <std::integral... Is>
	float&
	operator[](Is... dims)
	{
		const std::uint64_t dims_pack[]   = { static_cast<std::uint64_t>(dims)... };
		const std::uint64_t num_arguments = sizeof...(dims);
		const std::uint64_t index         = calculate_index(dims_pack, num_arguments);

		return __ptr[index];
	}

	template <std::integral... Is>
	float
	operator[](Is... dims) const
	{
		const std::uint64_t dims_pack[]   = { static_cast<std::uint64_t>(dims)... };
		const std::uint64_t num_arguments = sizeof...(dims);
		const std::uint64_t index         = calculate_index(dims_pack, num_arguments);

		return __ptr[index];
	}

public:
	static Tensor
	matmul(
		const Tensor& lhs,
		const Tensor& rhs);

	const Device&
	device() const noexcept
	{
		return __device;
	}

	const Shape&
	shape() const noexcept
	{
		return __shape;
	}

private:
	std::uint64_t
	calculate_index(
		const std::uint64_t dims_pack[],
		const std::uint64_t num_arguments) const;

private:
	float* __ptr    = nullptr;
	Device __device = {};
	Shape  __shape  = {};
};

}

#endif
