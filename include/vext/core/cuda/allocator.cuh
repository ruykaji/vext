#ifndef __VEXT_ALLOCATOR_CUDA_HPP__
#define __VEXT_ALLOCATOR_CUDA_HPP__

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                                            \
	do                                                                                                               \
		{                                                                                                             \
			cudaError_t err = (call);                                                                                  \
			if(err != cudaSuccess)                                                                                     \
				{                                                                                                       \
					std::cerr << __FILE__ << ":" << __LINE__ << " CUDA Error: " << cudaGetErrorString(err) << std::endl; \
					std::exit(EXIT_FAILURE);                                                                             \
				}                                                                                                       \
		}                                                                                                             \
	while(0)

namespace vext::core::cuda::allocator::kernel
{

constexpr std::uint64_t ALIGNMENT            = 256;
constexpr std::uint64_t SIZE_THRESHOLD       = 4 * 1024 * 1024;  /** 4MB */
constexpr std::uint64_t SMALL_POOL_BOUNDARY  = 4 * 512;          /** 2048B */
constexpr std::uint64_t LARGE_POOL_BOUNDARY  = 4 * 512 * 1024;   /** 2048KB */
constexpr std::uint64_t SMALL_POOL_SLAB_SIZE = 8 * 1024 * 1024;  /** 8MB */
constexpr std::uint64_t LARGE_POOL_SLAB_SIZE = 80 * 1024 * 1024; /** 80MB */

struct Block
{
	void*         ptr    = nullptr;
	std::uint64_t size   = 0;
	bool          in_use = false;

	Block* prev = nullptr;
	Block* next = nullptr;

	friend bool
	operator<(
		const Block& lhs,
		const Block& rhs)
	{
		return lhs.size == rhs.size ? lhs.ptr < rhs.ptr : lhs.size < rhs.size;
	}
};

template <typename Tp>
struct DereferenceCompareLess
{
	bool
	operator()(
		const Tp* lhs,
		const Tp* rhs) const
	{
		return (*lhs) < (*rhs);
	}
};

struct Pool
{
	std::set<Block*, DereferenceCompareLess<Block>> free_blocks      = {};
	std::unordered_map<const void*, Block*>         allocated_blocks = {};
	std::vector<Block*>                             roots            = {};
};

inline std::uint64_t
align_up(
	const std::uint64_t x,
	const std::uint64_t a)
{
	return ((x + a - 1) / a) * a;
}

inline std::pair<Block*, Block*>
maybe_split_block(
	Block*              block,
	const std::uint64_t requested_size)
{
	const std::uint64_t boundary = block->size <= SMALL_POOL_SLAB_SIZE ? SMALL_POOL_BOUNDARY : LARGE_POOL_BOUNDARY;

	Block* remainder = nullptr;

	if(const std::uint64_t diff = block->size - requested_size; diff >= boundary)
		{
			remainder = new Block{
				.ptr    = static_cast<char*>(block->ptr) + requested_size,
				.size   = diff,
				.in_use = false,
				.prev   = block,
				.next   = block->next,
			};

			if(block->next)
				{
					block->next->prev = remainder;
				}

			block->next = remainder;
			block->size = requested_size;
		}

	return { block, remainder };
}

inline Pool small_pool = {};
inline Pool large_pool = {};

}

namespace vext::core::cuda::allocator
{

template <typename Tp>
inline Tp*
allocate(
	const std::uint64_t count)
{
	if(count == 0)
		{
			throw std::invalid_argument("Cannot allocate zero bytes");
		}

	const std::uint64_t requested_size = count * sizeof(Tp);
	const std::uint64_t aligned_size   = kernel::align_up(requested_size, kernel::ALIGNMENT);

	void*         ptr          = nullptr;
	kernel::Pool& pool         = aligned_size < kernel::SIZE_THRESHOLD ? kernel::small_pool : kernel::large_pool;
	kernel::Block search_block = { .size = aligned_size };

	if(const auto it = pool.free_blocks.lower_bound(&search_block); it != pool.free_blocks.end())
		{
			kernel::Block* old_block = *it;
			pool.free_blocks.erase(it);

			const auto [used_block, remainder_block] = kernel::maybe_split_block(old_block, aligned_size);

			ptr                = used_block->ptr;
			used_block->in_use = true;

			if(remainder_block != nullptr)
				{
					pool.free_blocks.insert(used_block->next);
				}

			pool.allocated_blocks[ptr] = used_block;
		}
	else
		{
			const std::uint64_t alloc_size = aligned_size < kernel::SIZE_THRESHOLD ? kernel::SMALL_POOL_SLAB_SIZE : std::max(kernel::LARGE_POOL_SLAB_SIZE, kernel::align_up(aligned_size, kernel::LARGE_POOL_BOUNDARY));

			kernel::Block* new_block = new kernel::Block{ .size = alloc_size };

			if(const cudaError_t err = cudaMalloc(&new_block->ptr, alloc_size); err != cudaSuccess)
				{
					std::cerr << __FILE__ << ":" << __LINE__ << " CUDA Error: " << cudaGetErrorString(err) << std::endl;
					exit(EXIT_FAILURE);
				}

			const auto [used_block, remainder_block] = kernel::maybe_split_block(new_block, aligned_size);

			ptr                = used_block->ptr;
			used_block->in_use = true;

			if(remainder_block != nullptr)
				{
					pool.free_blocks.insert(used_block->next);
				}

			pool.allocated_blocks[ptr] = used_block;
			pool.roots.emplace_back(used_block);
		}

	if(const cudaError_t err = cudaMemset(ptr, 0, aligned_size); err != cudaSuccess)
		{
			std::cerr << __FILE__ << ":" << __LINE__ << " CUDA Error: " << cudaGetErrorString(err) << std::endl;
			exit(EXIT_FAILURE);
		}

	return static_cast<Tp*>(ptr);
}

inline void
deallocate(
	void* ptr)
{
	if(ptr == nullptr)
		{
			return;
		}

	kernel::Block* block = nullptr;
	kernel::Pool*  pool  = nullptr;

	if(const auto it = kernel::small_pool.allocated_blocks.find(ptr); it != kernel::small_pool.allocated_blocks.end())
		{
			block = it->second;
			pool  = &kernel::small_pool;
		}
	else if(const auto it = kernel::large_pool.allocated_blocks.find(ptr); it != kernel::large_pool.allocated_blocks.end())
		{
			block = it->second;
			pool  = &kernel::large_pool;
		}
	else
		{
			std::cerr << __FILE__ << ":" << __LINE__ << " CUDA Error: Failed to free allocated pointer" << std::endl;
			exit(EXIT_FAILURE);
		}

	kernel::Block* prev_block = block->prev;
	kernel::Block* next_block = block->next;
	kernel::Block* merged     = block;

	merged->in_use = false;

	if(prev_block != nullptr && !prev_block->in_use)
		{
			pool->free_blocks.erase(prev_block);

			prev_block->size = prev_block->size + merged->size;
			prev_block->next = merged->next;

			if(merged->next != nullptr)
				{
					merged->next->prev = prev_block;
				}

			delete merged;
			merged = prev_block;
		}

	if(next_block != nullptr && !next_block->in_use)
		{
			pool->free_blocks.erase(next_block);

			merged->size = merged->size + next_block->size;
			merged->next = next_block->next;

			if(next_block->next != nullptr)
				{
					next_block->next->prev = merged;
				}

			delete next_block;
		}

	pool->free_blocks.insert(merged);
	pool->allocated_blocks.erase(ptr);
}

inline void
free()
{
	for(auto& block : kernel::small_pool.roots)
		{
			CUDA_CHECK(cudaFree(block->ptr));

			while(block != nullptr)
				{
					kernel::Block* tmp = block;
					block              = block->next;

					delete tmp;
				}
		}

	kernel::small_pool.free_blocks.clear();
	kernel::small_pool.allocated_blocks.clear();
	kernel::small_pool.roots.clear();

	for(auto& block : kernel::large_pool.roots)
		{
			CUDA_CHECK(cudaFree(block->ptr));

			while(block != nullptr)
				{
					kernel::Block* tmp = block;
					block              = block->next;

					delete tmp;
				}
		}

	kernel::large_pool.free_blocks.clear();
	kernel::large_pool.allocated_blocks.clear();
	kernel::large_pool.roots.clear();
}

}

#undef CUDA_CHECK

#endif
