#ifndef __VEXT_SHAPE_HPP__
#define __VEXT_SHAPE_HPP__

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace vext
{

class Shape
{
public:
	using iterator       = std::vector<std::uint64_t>::iterator;
	using const_iterator = std::vector<std::uint64_t>::const_iterator;

	Shape(
		const std::initializer_list<std::uint64_t> dims)
		: __dims(dims)
	{
		compute_length();
		compute_strides();
	};

	explicit Shape(
		const std::vector<std::uint64_t>& dims)
		: __dims(dims)
	{
		compute_length();
		compute_strides();
	};

	explicit Shape(
		std::vector<std::uint64_t>&& dims)
		: __dims(std::forward<std::vector<std::uint64_t>>(dims))
	{
		compute_length();
		compute_strides();
	};

public:
	std::uint64_t
	operator[](
		const std::int64_t index) const noexcept
	{
		if(index >= 0)
			{
				return __dims[index];
			}

		return __dims[__dims.size() + index];
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
	static Shape
	broadcast(
		const Shape& source,
		const Shape& target);

	std::uint64_t
	at(
		const std::int64_t index) const
	{
		if(std::abs(index) >= __dims.size())
			{
				throw std::out_of_range("Attempt to out of range access in a Shape instance.");
			}

		if(index >= 0)
			{
				return __dims[index];
			}

		return __dims[__dims.size() + index];
	}

	std::uint64_t
	length() const noexcept
	{
		return __length;
	}

	std::uint64_t
	size() const noexcept
	{
		return __dims.size();
	}

	const std::vector<std::uint64_t>&
	dims() const noexcept
	{
		return __dims;
	}

	const std::vector<std::uint64_t>&
	strides() const noexcept
	{
		return __strides;
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
	std::uint64_t              __length  = 0;
	std::vector<std::uint64_t> __dims    = {};
	std::vector<std::uint64_t> __strides = {};
};

};

#endif
