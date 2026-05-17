#include <vext/shape.hpp>

namespace vext
{

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
	const std::size_t size = __dims.size();
	__strides.resize(size);

	if(size == 0)
		{
			return;
		}

	std::size_t stride = 1;

	for(std::size_t i = size; i > 0; --i)
		{
			__strides[i - 1] = stride;
			stride *= __dims[i - 1];
		}
}

}
