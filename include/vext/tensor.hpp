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

public:
	float*
	data() noexcept
	{
		return __ptr;
	}

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
	float* __ptr    = nullptr;
	Device __device = {};
	Shape  __shape  = {};
};

}

#endif
