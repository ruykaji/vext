#ifndef __VEXT_DEVICE_HPP__
#define __VEXT_DEVICE_HPP__

#include <vext/type.hpp>

namespace vext
{

class Device
{
public:
	Device(
		const type::Backend backend = type::Backend::CPU,
		const std::uint64_t index   = 0)
		: __backend(backend),
		  __index(index) {};

public:
	friend bool
	operator==(
		const Device& a,
		const Device& b)
	{
		return a.__backend == b.__backend && a.__index == b.__index;
	}

	friend bool
	operator!=(
		const Device& a,
		const Device& b)
	{
		return !(a == b);
	}

public:
	static Device
	cpu()
	{
		return { type::Backend::CPU, 0 };
	}

	static Device
	cuda(const std::uint64_t index = 0)
	{
		return { type::Backend::CUDA, index };
	}

	bool
	is_cpu() const
	{
		return __backend == type::Backend::CPU;
	}

	bool
	is_cuda() const
	{
		return __backend == type::Backend::CUDA;
	}

private:
	type::Backend __backend = type::Backend::CPU;
	std::uint64_t __index   = 0;
};

}

#endif
