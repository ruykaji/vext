#ifndef __VEXT_SHAPE_HPP__
#define __VEXT_SHAPE_HPP__

#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace vext
{

class Shape
{
public:
	using iterator       = std::vector<std::size_t>::iterator;
	using const_iterator = std::vector<std::size_t>::const_iterator;

	Shape(
		const std::initializer_list<std::size_t> dims)
		: __dims(dims) {};

	explicit Shape(
		const std::vector<std::size_t>& dims)
		: __dims() {};

public:
	std::size_t
	operator[](
		const std::size_t index) const noexcept
	{
		return __dims[index];
	}

	friend bool
	operator==(
		const Shape& lhs,
		const Shape& rhs) noexcept
	{
		return lhs.__dims == rhs.__dims;
	}

	friend bool
	operator!=(
		const Shape& lhs,
		const Shape& rhs) noexcept
	{
		return !(lhs == rhs);
	}

public:
	std::size_t
	at(
		const std::size_t index) const
	{
		if(index >= __dims.size())
			{
				throw std::out_of_range("Attempt to out of range access in a Shape instance.");
			}

		return __dims[index];
	}

	std::size_t
	length() const noexcept
	{
		return __length;
	}

	std::size_t
	size() const noexcept
	{
		return __dims.size();
	}

	iterator
	begin() noexcept
	{
		return __dims.begin();
	}

	iterator
	end() noexcept
	{
		return __dims.end();
	}

	const_iterator
	begin() const noexcept
	{
		return __dims.begin();
	}

	const_iterator
	end() const noexcept
	{
		return __dims.end();
	}

private:
	void
	compute_length() noexcept;

	void
	compute_strides() noexcept;

private:
	std::size_t              __length  = 0;
	std::vector<std::size_t> __dims    = {};
	std::vector<std::size_t> __strides = {};
};

};

#endif
