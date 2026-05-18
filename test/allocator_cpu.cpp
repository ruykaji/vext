#include <cstdint>
#include <cstring>
#include <stdexcept>

#include <gtest/gtest.h>

#include "cpu/allocator.hpp"

constexpr std::size_t ALIGNMENT           = 64;
constexpr std::size_t MIN_LARGE_POOL_SIZE = 20 * 1024 * 1024;
constexpr std::size_t SIZE_THRESHOLD      = 1024 * 1024;

class AllocatorCPUTest
	: public testing::Test
{
protected:
	void
	TearDown() override
	{
		vext::cpu::Allocator::free();
	}

	bool
	is_aligned(
		const void*       ptr,
		const std::size_t alignment)
	{
		return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
	}
};

TEST_F(AllocatorCPUTest, AllocateReturnsAlignedWritableMemory)
{
	void* ptr = vext::cpu::Allocator::allocate(1);

	ASSERT_NE(ptr, nullptr);
	EXPECT_TRUE(is_aligned(ptr, ALIGNMENT));

	std::memset(ptr, 0x2a, 1);

	vext::cpu::Allocator::deallocate(ptr);
}

TEST_F(AllocatorCPUTest, AllocateAlignsRequestedSizeBeforeSplitting)
{
	void* first  = vext::cpu::Allocator::allocate(65);
	void* second = vext::cpu::Allocator::allocate(1);

	ASSERT_NE(first, nullptr);
	ASSERT_NE(second, nullptr);

	EXPECT_TRUE(is_aligned(first, ALIGNMENT));
	EXPECT_TRUE(is_aligned(second, ALIGNMENT));
	EXPECT_EQ(static_cast<char*>(second) - static_cast<char*>(first), 128);

	vext::cpu::Allocator::deallocate(second);
	vext::cpu::Allocator::deallocate(first);
}

TEST_F(AllocatorCPUTest, ReusesFreedSmallPoolBlock)
{
	void* first = vext::cpu::Allocator::allocate(32);
	vext::cpu::Allocator::deallocate(first);

	void* second = vext::cpu::Allocator::allocate(32);

	EXPECT_EQ(second, first);

	vext::cpu::Allocator::deallocate(second);
}

TEST_F(AllocatorCPUTest, CoalescesPreviousAndNextFreeBlocks)
{
	void* first  = vext::cpu::Allocator::allocate(64);
	void* second = vext::cpu::Allocator::allocate(64);
	void* third  = vext::cpu::Allocator::allocate(64);

	vext::cpu::Allocator::deallocate(first);
	vext::cpu::Allocator::deallocate(third);
	vext::cpu::Allocator::deallocate(second);

	void* merged = vext::cpu::Allocator::allocate(64);

	EXPECT_EQ(merged, first);

	vext::cpu::Allocator::deallocate(merged);
}

TEST_F(AllocatorCPUTest, AllocatesFromSeparateSmallAndLargePools)
{
	void* small = vext::cpu::Allocator::allocate(512);
	void* large = vext::cpu::Allocator::allocate(1024 * 1024);

	ASSERT_NE(small, nullptr);
	ASSERT_NE(large, nullptr);

	EXPECT_TRUE(is_aligned(small, ALIGNMENT));
	EXPECT_TRUE(is_aligned(large, ALIGNMENT));
	EXPECT_NE(small, large);

	vext::cpu::Allocator::deallocate(large);
	vext::cpu::Allocator::deallocate(small);
}

TEST_F(AllocatorCPUTest, HandlesLargePoolAllocationWithoutRemainderSplit)
{
	void* ptr = vext::cpu::Allocator::allocate(MIN_LARGE_POOL_SIZE);

	ASSERT_NE(ptr, nullptr);
	EXPECT_TRUE(is_aligned(ptr, ALIGNMENT));

	std::memset(ptr, 0x3f, MIN_LARGE_POOL_SIZE);

	vext::cpu::Allocator::deallocate(ptr);
}

TEST_F(AllocatorCPUTest, SplitsAndReusesLargePoolBlocks)
{
	void* first  = vext::cpu::Allocator::allocate(SIZE_THRESHOLD);
	void* second = vext::cpu::Allocator::allocate(SIZE_THRESHOLD);

	ASSERT_NE(first, nullptr);
	ASSERT_NE(second, nullptr);

	EXPECT_EQ(static_cast<char*>(second) - static_cast<char*>(first), SIZE_THRESHOLD);

	vext::cpu::Allocator::deallocate(second);
	vext::cpu::Allocator::deallocate(first);
}

TEST_F(AllocatorCPUTest, DeallocateRejectsUnknownPointers)
{
	std::int32_t value = 0;

	EXPECT_THROW(vext::cpu::Allocator::deallocate(&value), std::invalid_argument);
	EXPECT_THROW(vext::cpu::Allocator::deallocate(nullptr), std::invalid_argument);
}

TEST_F(AllocatorCPUTest, DeallocateRejectsDoubleFree)
{
	void* ptr = vext::cpu::Allocator::allocate(64);

	vext::cpu::Allocator::deallocate(ptr);

	EXPECT_THROW(vext::cpu::Allocator::deallocate(ptr), std::invalid_argument);
}

TEST_F(AllocatorCPUTest, FreeReleasesAllPoolsAndAllowsFreshAllocation)
{
	void* small = vext::cpu::Allocator::allocate(256);
	void* large = vext::cpu::Allocator::allocate(1024 * 1024);

	ASSERT_NE(small, nullptr);
	ASSERT_NE(large, nullptr);

	vext::cpu::Allocator::free();

	void* next = vext::cpu::Allocator::allocate(256);

	ASSERT_NE(next, nullptr);
	EXPECT_TRUE(is_aligned(next, ALIGNMENT));

	vext::cpu::Allocator::deallocate(next);
}

int
main(int argc, char** argv)
{
	testing::InitGoogleTest();
	return RUN_ALL_TESTS();
}
