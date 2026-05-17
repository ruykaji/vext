#include <cstdint>
#include <cstring>
#include <stdexcept>

#include <gtest/gtest.h>

#include "cpu/allocator.hpp"

namespace
{

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
		vext::AllocatorCPU::free();
	}
};

bool
is_aligned(
	const void*       ptr,
	const std::size_t alignment)
{
	return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

}

TEST_F(AllocatorCPUTest, AllocateReturnsAlignedWritableMemory)
{
	void* ptr = vext::AllocatorCPU::allocate(1);

	ASSERT_NE(ptr, nullptr);
	EXPECT_TRUE(is_aligned(ptr, ALIGNMENT));

	std::memset(ptr, 0x2a, 1);

	vext::AllocatorCPU::deallocate(ptr);
}

TEST_F(AllocatorCPUTest, AllocateAlignsRequestedSizeBeforeSplitting)
{
	void* first  = vext::AllocatorCPU::allocate(65);
	void* second = vext::AllocatorCPU::allocate(1);

	ASSERT_NE(first, nullptr);
	ASSERT_NE(second, nullptr);

	EXPECT_TRUE(is_aligned(first, ALIGNMENT));
	EXPECT_TRUE(is_aligned(second, ALIGNMENT));
	EXPECT_EQ(static_cast<char*>(second) - static_cast<char*>(first), 128);

	vext::AllocatorCPU::deallocate(second);
	vext::AllocatorCPU::deallocate(first);
}

TEST_F(AllocatorCPUTest, ReusesFreedSmallPoolBlock)
{
	void* first = vext::AllocatorCPU::allocate(32);
	vext::AllocatorCPU::deallocate(first);

	void* second = vext::AllocatorCPU::allocate(32);

	EXPECT_EQ(second, first);

	vext::AllocatorCPU::deallocate(second);
}

TEST_F(AllocatorCPUTest, CoalescesPreviousAndNextFreeBlocks)
{
	void* first  = vext::AllocatorCPU::allocate(64);
	void* second = vext::AllocatorCPU::allocate(64);
	void* third  = vext::AllocatorCPU::allocate(64);

	vext::AllocatorCPU::deallocate(first);
	vext::AllocatorCPU::deallocate(third);
	vext::AllocatorCPU::deallocate(second);

	void* merged = vext::AllocatorCPU::allocate(64);

	EXPECT_EQ(merged, first);

	vext::AllocatorCPU::deallocate(merged);
}

TEST_F(AllocatorCPUTest, AllocatesFromSeparateSmallAndLargePools)
{
	void* small = vext::AllocatorCPU::allocate(512);
	void* large = vext::AllocatorCPU::allocate(1024 * 1024);

	ASSERT_NE(small, nullptr);
	ASSERT_NE(large, nullptr);

	EXPECT_TRUE(is_aligned(small, ALIGNMENT));
	EXPECT_TRUE(is_aligned(large, ALIGNMENT));
	EXPECT_NE(small, large);

	vext::AllocatorCPU::deallocate(large);
	vext::AllocatorCPU::deallocate(small);
}

TEST_F(AllocatorCPUTest, HandlesLargePoolAllocationWithoutRemainderSplit)
{
	void* ptr = vext::AllocatorCPU::allocate(MIN_LARGE_POOL_SIZE);

	ASSERT_NE(ptr, nullptr);
	EXPECT_TRUE(is_aligned(ptr, ALIGNMENT));

	std::memset(ptr, 0x3f, MIN_LARGE_POOL_SIZE);

	vext::AllocatorCPU::deallocate(ptr);
}

TEST_F(AllocatorCPUTest, SplitsAndReusesLargePoolBlocks)
{
	void* first  = vext::AllocatorCPU::allocate(SIZE_THRESHOLD);
	void* second = vext::AllocatorCPU::allocate(SIZE_THRESHOLD);

	ASSERT_NE(first, nullptr);
	ASSERT_NE(second, nullptr);

	EXPECT_EQ(static_cast<char*>(second) - static_cast<char*>(first), SIZE_THRESHOLD);

	vext::AllocatorCPU::deallocate(second);
	vext::AllocatorCPU::deallocate(first);
}

TEST_F(AllocatorCPUTest, DeallocateRejectsUnknownPointers)
{
	std::int32_t value = 0;

	EXPECT_THROW(vext::AllocatorCPU::deallocate(&value), std::invalid_argument);
	EXPECT_THROW(vext::AllocatorCPU::deallocate(nullptr), std::invalid_argument);
}

TEST_F(AllocatorCPUTest, DeallocateRejectsDoubleFree)
{
	void* ptr = vext::AllocatorCPU::allocate(64);

	vext::AllocatorCPU::deallocate(ptr);

	EXPECT_THROW(vext::AllocatorCPU::deallocate(ptr), std::invalid_argument);
}

TEST_F(AllocatorCPUTest, FreeReleasesAllPoolsAndAllowsFreshAllocation)
{
	void* small = vext::AllocatorCPU::allocate(256);
	void* large = vext::AllocatorCPU::allocate(1024 * 1024);

	ASSERT_NE(small, nullptr);
	ASSERT_NE(large, nullptr);

	vext::AllocatorCPU::free();

	void* next = vext::AllocatorCPU::allocate(256);

	ASSERT_NE(next, nullptr);
	EXPECT_TRUE(is_aligned(next, ALIGNMENT));

	vext::AllocatorCPU::deallocate(next);
}

int
main(int argc, char** argv)
{
	testing::InitGoogleTest();
	return RUN_ALL_TESTS();
}
