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

	template <std::integral... Is>
	Shape(
		Is... dims)
	{
		__dims = { static_cast<std::uint64_t>(dims)... };

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
		const Shape& target)
	{
		const std::uint64_t source_size = source.size();
		const std::uint64_t target_size = target.size();

		if(target_size > source_size)
			{
				throw std::runtime_error("");
			}

		std::uint64_t offset_left = 0;
		std::uint64_t subset_size = 0;

		for(std::uint64_t i = 0; i < source_size; ++i)
			{
				if(source.__dims[i] == target.__dims[subset_size])
					{
						++subset_size;

						if(subset_size == target_size)
							{
								break;
							}

						continue;
					}

				subset_size = 0;
				offset_left = i + 1;
			}

		if(subset_size == 0 && target.__length != 1)
			{
				throw std::runtime_error("");
			}

		std::vector<std::uint64_t> dims;
		dims.resize(source_size, 1);

		std::copy(target.__dims.begin(), target.__dims.end(), dims.begin() + offset_left);

		Shape shape(std::move(dims));

		for(std::uint64_t i = 0; i < offset_left; ++i)
			{
				shape.__strides[i] = 0;
			}

		for(std::uint64_t i = offset_left + target_size; i < source_size; ++i)
			{
				shape.__strides[i] = 0;
			}

		return shape;
	}

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
	compute_length() noexcept
	{
		__length = 1;

		for(const auto dim : __dims)
			{
				__length *= dim;
			}
	}

	void
	compute_strides() noexcept
	{
		const std::uint64_t size = __dims.size();
		__strides.resize(size);

		if(size == 0)
			{
				return;
			}

		std::uint64_t stride = 1;

		for(std::uint64_t i = size; i > 0; --i)
			{
				__strides[i - 1] = stride;
				stride *= __dims[i - 1];
			}
	}

private:
	std::uint64_t              __length  = 0;
	std::vector<std::uint64_t> __dims    = {};
	std::vector<std::uint64_t> __strides = {};
};

};

#endif
