#include <cstdlib>
#include <cstring>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "allocator.hpp"

namespace vext::cpu::allocator
{

constexpr std::size_t ALIGNMENT           = 64;
constexpr std::size_t SIZE_THRESHOLD      = 1024 * 1024;      /** 1MB */
constexpr std::size_t SMALL_POOL_BOUNDARY = 512;              /** 512B */
constexpr std::size_t LARGE_POOL_BOUNDARY = 512 * 1024;       /** 512KB */
constexpr std::size_t MIN_SMALL_POOL_SIZE = 2 * 1024 * 1024;  /** 2MB */
constexpr std::size_t MIN_LARGE_POOL_SIZE = 20 * 1024 * 1024; /** 20MB */

template <typename Tp, bool Less = true>
struct DereferenceCompare
{
	bool
	operator()(
		const Tp* lhs,
		const Tp* rhs) const
	{
		if constexpr(Less)
			{
				return (*lhs) < (*rhs);
			}
		else
			{
				return (*lhs) > (*rhs);
			}
	}
};

struct Block
{
	void*       ptr    = nullptr;
	std::size_t size   = 0;
	bool        in_use = false;

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

struct Pool

{
	std::set<Block*, DereferenceCompare<Block>> free_blocks      = {};
	std::unordered_map<const void*, Block*>     allocated_blocks = {};
	std::vector<Block*>                         roots            = {};
};

static std::size_t
size(
	const std::size_t requested_size)
{
	if(requested_size < SIZE_THRESHOLD)
		{
			return MIN_SMALL_POOL_SIZE;
		}

	if(requested_size < MIN_LARGE_POOL_SIZE)
		{
			return MIN_LARGE_POOL_SIZE;
		}

	return ((requested_size + LARGE_POOL_BOUNDARY - 1) / LARGE_POOL_BOUNDARY) * LARGE_POOL_BOUNDARY;
}

static Block*
split_block(
	Block*            block,
	const std::size_t requested_size)
{
	const std::size_t boundary = requested_size < SIZE_THRESHOLD ? SMALL_POOL_BOUNDARY : LARGE_POOL_BOUNDARY;

	if(const std::size_t diff = block->size - requested_size; diff >= boundary)
		{
			Block* new_block = new Block{
				.ptr    = static_cast<char*>(block->ptr) + requested_size,
				.size   = diff,
				.in_use = false,
				.prev   = block,
				.next   = block->next,
			};

			if(block->next)
				{
					block->next->prev = new_block;
				}

			block->next = new_block;
			block->size = requested_size;
		}

	return block;
}

}

namespace vext::cpu
{

static allocator::Pool small_pool = {};
static allocator::Pool large_pool = {};

void*
Allocator::allocate(
	const std::size_t requested_size)
{
	const std::size_t aligned_size = ((requested_size + allocator::ALIGNMENT - 1) / allocator::ALIGNMENT) * allocator::ALIGNMENT;

	void*            ptr          = nullptr;
	allocator::Pool& pool         = aligned_size < allocator::SIZE_THRESHOLD ? small_pool : large_pool;
	allocator::Block search_block = { .size = aligned_size };

	if(const auto it = pool.free_blocks.lower_bound(&search_block); it != pool.free_blocks.end())
		{
			allocator::Block* old_block = *it;
			pool.free_blocks.erase(it);

			allocator::Block* split_block = allocator::split_block(old_block, aligned_size);
			ptr                           = split_block->ptr;
			split_block->in_use           = true;

			if(split_block->next != nullptr)
				{
					pool.free_blocks.insert(split_block->next);
				}

			pool.allocated_blocks[ptr] = split_block;
		}
	else
		{
			const std::size_t allocation_size = allocator::size(aligned_size);

			allocator::Block* new_block = new allocator::Block{
				.size = allocation_size,
			};

			if(posix_memalign(&new_block->ptr, allocator::ALIGNMENT, allocation_size) != 0)
				{
					delete new_block;
					throw std::runtime_error("Failed to allocate memory");
				}

			allocator::Block* split_block = allocator::split_block(new_block, aligned_size);
			ptr                           = split_block->ptr;
			split_block->in_use           = true;

			if(split_block->next != nullptr)
				{
					pool.free_blocks.insert(split_block->next);
				}

			pool.allocated_blocks[ptr] = split_block;
			pool.roots.emplace_back(split_block);
		}

	std::memset(ptr, 0, requested_size);

	return ptr;
}

void
Allocator::deallocate(
	void* ptr)
{
	allocator::Block* block = nullptr;
	allocator::Pool*  pool  = nullptr;

	if(const auto it = small_pool.allocated_blocks.find(ptr); it != small_pool.allocated_blocks.end())
		{
			block = it->second;
			pool  = &small_pool;
		}
	else if(const auto it = large_pool.allocated_blocks.find(ptr); it != large_pool.allocated_blocks.end())
		{
			block = it->second;
			pool  = &large_pool;
		}
	else
		{
			throw std::invalid_argument("Failed to find allocated pointer");
		}

	allocator::Block* prev_block = block->prev;
	allocator::Block* next_block = block->next;
	allocator::Block* merged     = block;
	merged->in_use               = false;

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

void
Allocator::free()
{
	for(auto& block : small_pool.roots)
		{
			std::free(block->ptr);

			while(block != nullptr)
				{
					allocator::Block* tmp = block;
					block                 = block->next;

					delete tmp;
				}
		}

	small_pool.free_blocks.clear();
	small_pool.allocated_blocks.clear();
	small_pool.roots.clear();

	for(auto& block : large_pool.roots)
		{
			std::free(block->ptr);

			while(block != nullptr)
				{
					allocator::Block* tmp = block;
					block                 = block->next;

					delete tmp;
				}
		}

	large_pool.free_blocks.clear();
	large_pool.allocated_blocks.clear();
	large_pool.roots.clear();
}

}
