#include <vext/shape.hpp>

namespace vext
{

Shape
Shape::broadcast(
	const Shape& source,
	const Shape& target)
{
	const std::uint64_t source_size = source.size();
	const std::uint64_t target_size = target.size();

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
				}
			else
				{
					subset_size = 0;
					offset_left = i + 1;

				}
		}

	if(subset_size == 0)
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

void
Shape::compute_length() noexcept
{
	__length = 1;

	for(const auto dim : __dims)
		{
			__length *= dim;
		}
}

void
Shape::compute_strides() noexcept
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

}
