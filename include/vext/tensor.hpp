#ifndef __VEXT_TENSOR_HPP__
#define __VEXT_TENSOR_HPP__

#include <initializer_list>
#include <queue>
#include <stdexcept>
#include <utility>

#include <vext/core/cpu/allocator.hpp>
#include <vext/core/cpu/ops/memory.hpp>

#if VEXT_CUDA
#include <vext/core/cuda/allocator.cuh>
#include <vext/core/cuda/ops/memory.cuh>
#endif

#include <vext/core/type.hpp>
#include <vext/type.hpp>

namespace vext
{

template <typename T1, Backend B1 = Backend::CPU>
class Tensor
{
	template <typename T2, Backend B2>
	friend class Tensor;

	template <typename Tp>
	struct initializer_dimension
	{
		Tp                                 value    = 0;
		std::vector<initializer_dimension> children = {};

		initializer_dimension(
			const Tp value)
			: value(value) {};

		initializer_dimension(
			const std::initializer_list<initializer_dimension>& children)
			: children(children) {};
	};

public:
	using value_type                      = T1;
	static constexpr Backend backend_type = B1;

public:
	template <std::integral... Is>
	explicit Tensor(
		Is... dims)
		: __dims({ static_cast<std::uint32_t>(dims)... })
	{
		if constexpr(sizeof...(dims) == 0)
			{
				return;
			}

		compute_shape();
		allocate();
	}

	template <std::integral... Is>
	explicit Tensor(
		const std::vector<std::uint32_t>& dims)
		: __dims(dims)
	{
		compute_shape();
		allocate();
	}

	explicit Tensor(
		const std::initializer_list<initializer_dimension<T1>>& list)
	{
		if(list.size() == 0)
			{
				throw std::runtime_error("Tensor initializer list must contain at least one element.");
			}

		const initializer_dimension<T1> root(list);

		std::vector<std::uint32_t> dims;
		std::vector<T1>            data;

		for(initializer_dimension<T1> const* node = &root; !node->children.empty(); node = &node->children[0])
			{
				const std::uint64_t size = node->children.size();

				if(size == 1)
					{
						continue;
					}

				dims.emplace_back(size);
			}

		std::queue<const initializer_dimension<T1>*> queue;
		queue.emplace(&root);

		while(!queue.empty())
			{
				const initializer_dimension<T1>* node = queue.front();
				queue.pop();

				const std::vector<initializer_dimension<T1>>& children = node->children;

				if(children.empty())
					{
						throw std::invalid_argument("Tensor initializer contains an empty nested dimension; every dimension must contain at least one element.");
					}

				const std::uint64_t expected_size = children.front().children.size();

				for(const auto& child : children)
					{
						if(child.children.size() != expected_size)
							{
								throw std::invalid_argument("Tensor initializer is ragged; every nested list at the same depth must have the same length.");
							}

						if(child.children.empty())
							{
								data.emplace_back(child.value);
							}
						else
							{
								queue.emplace(&child);
							}
					}
			}

		if(dims.empty() && data.size() == 1)
			{
				__dims = { 1 };
			}
		else
			{
				__dims = std::move(dims);
			}

		compute_shape();
		allocate<B1>(data.data());
	}

	template <Backend B2>
	Tensor(
		const Tensor<T1, B2>& other)
	{
		__length  = other.__length;
		__dims    = other.__dims;
		__strides = other.__strides;

		allocate<B2>(other.__ptr);
	}

	Tensor(
		const Tensor& other)
	{
		__length  = other.__length;
		__dims    = other.__dims;
		__strides = other.__strides;

		allocate<B1>(other.__ptr);
	}

	Tensor(
		Tensor&& other)
	{
		__length  = std::exchange(other.__length, core::MIN_LENGTH);
		__dims    = std::exchange(other.__dims, { core::MIN_LENGTH });
		__strides = std::exchange(other.__strides, {});
		__ptr     = std::exchange(other.__ptr, nullptr);
	}

	~Tensor()
	{
		deallocate();
	}

public:
	template <Backend B2>
	Tensor&
	operator=(
		const Tensor<T1, B2>& other)
	{
		if(this != &other)
			{
				deallocate();

				__length  = other.__length;
				__dims    = other.__dims;
				__strides = other.__strides;

				allocate<B1>(other.__ptr);
			}

		return *this;
	}

	Tensor&
	operator=(
		const Tensor& other)
	{
		if(this != &other)
			{
				deallocate();

				__length  = other.__length;
				__dims    = other.__dims;
				__strides = other.__strides;

				allocate<B1>(other.__ptr);
			}

		return *this;
	}

	Tensor&
	operator=(
		Tensor&& other)
	{
		if(this != &other)
			{
				deallocate();

				__length  = std::exchange(other.__length, core::MIN_LENGTH);
				__dims    = std::exchange(other.__dims, { core::MIN_LENGTH });
				__strides = std::exchange(other.__strides, {});

				__ptr = std::exchange(other.__ptr, nullptr);
			}

		return *this;
	}

public:
	void
	set_from(
		const std::vector<T1>& values)
	{
		if constexpr(B1 == Backend::CPU)
			{
				core::cpu::ops::memcpy<T1>(__ptr, values.data(), std::min<std::uint32_t>(values.size(), __length));
			}
		#if VEXT_CUDA
		else
			{
				core::cuda::ops::memcpy<T1, B1, Backend::CPU>(__ptr, values.data(), std::min<std::uint32_t>(values.size(), __length));
			}
		#else
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
		#endif
	}

	template <std::integral... Is>
	T1
	item(
		Is... dims) const
	{
		std::uint32_t index = 0;

		if constexpr(sizeof...(dims) == 1)
			{
				index = static_cast<std::uint32_t>((dims, ...));
			}
		else if constexpr(sizeof...(dims) > 1)
			{
				index = flat_index(dims...);

				if(index >= __length)
					{
						throw std::invalid_argument("Computed tensor index is outside the allocated storage.");
					}
			}

		if constexpr(B1 == Backend::CPU)
			{
				return __ptr[index];
			}
		#if VEXT_CUDA
		else
			{
				return core::cuda::ops::memget(__ptr, index);
			}
		#else
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
		#endif
	}

	template <std::integral... Is>
	void
	put(
		const T1 value,
		Is... dims)
	{
		std::uint32_t index = 0;

		if constexpr(sizeof...(dims) == 1)
			{
				index = static_cast<std::uint32_t>((dims, ...));
			}
		else if constexpr(sizeof...(dims) > 1)
			{
				index = flat_index(dims...);

				if(index >= __length)
					{
						throw std::invalid_argument("Computed tensor index is outside the allocated storage.");
					}
			}

		if constexpr(B1 == Backend::CPU)
			{
				__ptr[index] = value;
			}
		#if VEXT_CUDA
		else
			{
				return core::cuda::ops::memset(__ptr + index, value, 1);
			}
		#else
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
		#endif
	}

	T1*
	data() noexcept
	{
		return __ptr;
	}

	const T1*
	data() const noexcept
	{
		return __ptr;
	}

	std::uint32_t
	length() const noexcept
	{
		return __length;
	}

	const std::vector<std::uint32_t>&
	dims() const noexcept
	{
		return __dims;
	}

	const std::vector<std::uint32_t>&
	strides() const noexcept
	{
		return __strides;
	}

private:
	void
	compute_shape()
	{
		const std::uint64_t size = __dims.size();

		if(size < core::MIN_RANK)
			{
				throw std::runtime_error("Tensor rank must be at least one; the provided shape has no dimensions.");
			}

		if(size > core::MAX_RANK)
			{
				throw std::runtime_error("Tensor rank exceeds the maximum supported rank.");
			}

		__strides.resize(size, 0);
		__length = 1;

		std::uint32_t stride = 1;

		for(std::uint64_t i = size; i > 0; --i)
			{
				__length *= __dims[i - 1];

				if(__length > core::MAX_LENGTH)
					{
						throw std::overflow_error("Tensor element count exceeds the maximum supported length.");
					}

				__strides[i - 1] = stride;
				stride *= __dims[i - 1];
			}

		if(__length < core::MIN_LENGTH)
			{
				throw std::overflow_error("Tensor dimensions must be greater than zero; the computed element count is zero.");
			}
	}

	void
	allocate()
	{
		if constexpr(B1 == Backend::CPU)
			{
				__ptr = core::cpu::allocator::allocate<T1>(__length);
			}
		#if VEXT_CUDA
		else
			{
				__ptr = core::cuda::allocator::allocate<T1>(__length);
			}
		#else
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
		#endif
	}

	template <Backend B2>
	void
	allocate(
		T1* data)
	{
		if constexpr(B1 == Backend::CPU)
			{
				__ptr = core::cpu::allocator::allocate<T1>(__length);

				if constexpr(B2 == Backend::CPU)
					{
						core::cpu::ops::memcpy<T1>(__ptr, data, __length);
					}
				#if VEXT_CUDA
				else
					{
						core::cuda::ops::memcpy<T1, B1, B2>(__ptr, data, __length);
					}
				#else
				else
					{
						static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
					}
				#endif
			}
		#if VEXT_CUDA
		else
			{
				__ptr = core::cuda::allocator::allocate<T1>(__length);
				core::cuda::ops::memcpy<T1, B1, B2>(__ptr, data, __length);
			}
		#else
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
		#endif
	}

	void
	deallocate()
	{
		if constexpr(B1 == Backend::CPU)
			{
				core::cpu::allocator::deallocate(__ptr);
			}
		#if VEXT_CUDA
		else
			{
				core::cuda::allocator::deallocate(__ptr);
			}
		#else
		else
			{
				static_assert(core::dependent_false<B1>, "Unsupported backend or missing VEXT_CUDA flag.");
			}
		#endif
	}

	template <std::integral... Is>
	std::uint64_t
	flat_index(
		Is... dims) const
	{
		const std::uint32_t dims_pack[]   = { static_cast<std::uint32_t>(dims)... };
		const std::uint32_t num_arguments = sizeof...(dims);

		if(num_arguments != __dims.size())
			{
				throw std::runtime_error("Tensor index rank does not match the tensor rank.");
			}

		std::uint32_t index = 0;

		for(std::uint32_t i = 0; i < num_arguments; ++i)
			{
				if(dims_pack[i] >= __dims[i])
					{
						throw std::runtime_error("Tensor coordinate is outside the bounds of its dimension.");
					}

				index += dims_pack[i] * __strides[i];
			}

		if(index >= __length)
			{
				throw std::runtime_error("Computed tensor index is outside the allocated storage.");
			}

		return index;
	}

private:
	T1*                        __ptr     = nullptr;
	std::uint32_t              __length  = 0;
	std::vector<std::uint32_t> __dims    = { 0 };
	std::vector<std::uint32_t> __strides = {};
};

}

#endif
