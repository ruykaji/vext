#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include <vext/core/cpu/allocator.hpp>
#include <vext/core/cpu/operations/elementwise_binary.hpp>
#include <vext/core/cpu/operations/elementwise_logical.hpp>
#include <vext/core/cpu/operations/elementwise_unary.hpp>
#include <vext/core/cpu/operations/linear_algebra.hpp>
#include <vext/core/cpu/operations/memory.hpp>
#include <vext/core/cpu/operations/reduction.hpp>

namespace
{

template <typename Tp>
void
expect_vector_eq(
	const std::vector<Tp>& actual,
	const std::vector<Tp>& expected)
{
	ASSERT_EQ(actual.size(), expected.size());

	for(std::uint64_t i = 0; i < actual.size(); ++i)
		{
			EXPECT_EQ(actual[i], expected[i]) << "at index " << i;
		}
}

void
expect_vector_near(
	const std::vector<float>& actual,
	const std::vector<float>& expected,
	const float               tolerance = 1e-5f)
{
	ASSERT_EQ(actual.size(), expected.size());

	for(std::uint64_t i = 0; i < actual.size(); ++i)
		{
			EXPECT_NEAR(actual[i], expected[i], tolerance) << "at index " << i;
		}
}

}

TEST(CoreCpuAllocator, AllocatesAlignedZeroInitializedSmallAndLargeBlocks)
{
    std::uint8_t* small = vext::core::cpu::allocator::allocate<std::uint8_t>(257);
    std::uint8_t* large = vext::core::cpu::allocator::allocate<std::uint8_t>(1024 * 1024 + 17);

	ASSERT_NE(small, nullptr);
	ASSERT_NE(large, nullptr);
	EXPECT_EQ(reinterpret_cast<std::uintptr_t>(small) % vext::core::cpu::allocator::kernel::ALIGNMENT, 0);
	EXPECT_EQ(reinterpret_cast<std::uintptr_t>(large) % vext::core::cpu::allocator::kernel::ALIGNMENT, 0);

	for(std::uint64_t i = 0; i < 257; ++i)
		{
			EXPECT_EQ(small[i], 0);
		}

	vext::core::cpu::allocator::deallocate(small);
	vext::core::cpu::allocator::deallocate(large);
	vext::core::cpu::allocator::free();
}

TEST(CoreCpuAllocator, ReusesFreedSmallPoolBlocksThroughPublicAllocate)
{
	void* first = vext::core::cpu::allocator::allocate<std::uint8_t>(128);
	vext::core::cpu::allocator::deallocate(first);

	void* second = vext::core::cpu::allocator::allocate<std::uint8_t>(128);

	EXPECT_EQ(second, first);

	vext::core::cpu::allocator::deallocate(second);
	vext::core::cpu::allocator::free();
}

TEST(CoreCpuAllocator, RejectsZeroByteAllocation)
{
	EXPECT_THROW(vext::core::cpu::allocator::allocate<std::uint8_t>(0), std::invalid_argument);
}

TEST(CoreCpuMemory, CopiesAndSetsElementCounts)
{
	std::vector<std::int32_t> src{ 1, 2, 3, 4 };
	std::vector<std::int32_t> dst(4, 0);

	vext::core::cpu::operations::memcpy(dst.data(), src.data(), static_cast<std::uint32_t>(src.size()));
	expect_vector_eq(dst, src);

	vext::core::cpu::operations::memset(dst.data(), 7, static_cast<std::uint32_t>(dst.size()));
	expect_vector_eq(dst, (std::vector<std::int32_t>{ 0x07070707, 0x07070707, 0x07070707, 0x07070707 }));
}

TEST(CoreCpuBinaryOperations, ComputesSameShapeArithmetic)
{
	const std::vector<int> a{ 8, 12, 20 };
	const std::vector<int> b{ 2, 3, 4 };
	std::vector<int>       out(3);

	vext::core::cpu::operations::binary<vext::core::BinaryOperation::ADD>(out.data(), a.data(), b.data(), 3);
	expect_vector_eq(out, (std::vector<int>{ 10, 15, 24 }));

	vext::core::cpu::operations::binary<vext::core::BinaryOperation::SUB>(out.data(), a.data(), b.data(), 3);
	expect_vector_eq(out, (std::vector<int>{ 6, 9, 16 }));

	vext::core::cpu::operations::binary<vext::core::BinaryOperation::MUL>(out.data(), a.data(), b.data(), 3);
	expect_vector_eq(out, (std::vector<int>{ 16, 36, 80 }));

	vext::core::cpu::operations::binary<vext::core::BinaryOperation::DIV>(out.data(), a.data(), b.data(), 3);
	expect_vector_eq(out, (std::vector<int>{ 4, 4, 5 }));

	vext::core::cpu::operations::binary<vext::core::BinaryOperation::POW>(out.data(), b.data(), b.data(), 3);
	expect_vector_eq(out, (std::vector<int>{ 4, 27, 256 }));
}

TEST(CoreCpuBinaryOperations, BroadcastsWithSuppliedStrides)
{
	const std::vector<float> matrix{ 1, 2, 3, 4, 5, 6 };
	const std::vector<float> bias{ 10, 20, 30 };
	std::vector<float>       out(6);

	vext::core::cpu::operations::binary_with_broadcast<vext::core::BinaryOperation::ADD>(
		out.data(),
		matrix.data(),
		bias.data(),
		6,
		std::vector<std::uint32_t>{ 2, 3 },
		std::vector<std::uint32_t>{ 0, 1 });

	expect_vector_near(out, { 11, 22, 33, 14, 25, 36 });
}

TEST(CoreCpuBinaryOperations, ComputesPrelu)
{
	const std::vector<float> x{ -2.0f, -1.0f, 0.0f, 3.0f };
	const std::vector<float> slope{ 0.25f, 0.5f, 0.75f, 1.0f };
	std::vector<float>       out(4);

	vext::core::cpu::operations::binary<vext::core::BinaryOperation::PRELU>(out.data(), x.data(), slope.data(), 4);

	expect_vector_near(out, { -0.5f, -0.5f, 0.0f, 3.0f });
}

TEST(CoreCpuLogicalOperations, ProducesMaskBytes)
{
	const std::vector<int>    a{ 1, 2, 3, 4 };
	const std::vector<int>    b{ 1, 0, 3, 5 };
	std::vector<std::uint8_t> out(4);

	vext::core::cpu::operations::logical<vext::core::LogicOperation::EQUAL>(out.data(), a.data(), b.data(), 4);
	expect_vector_eq(out, (std::vector<std::uint8_t>{ 1, 0, 1, 0 }));

	vext::core::cpu::operations::logical<vext::core::LogicOperation::LESS>(out.data(), a.data(), b.data(), 4);
	expect_vector_eq(out, (std::vector<std::uint8_t>{ 0, 0, 0, 1 }));
}

TEST(CoreCpuUnaryOperations, MutatesValuesInPlace)
{
	std::vector<float> values{ -1.0f, 0.0f, 4.0f };

	vext::core::cpu::operations::unary<vext::core::UnaryOperation::ABS>(values.data(), 3);
	expect_vector_near(values, { 1.0f, 0.0f, 4.0f });

	vext::core::cpu::operations::unary<vext::core::UnaryOperation::SQRT>(values.data(), 3);
	expect_vector_near(values, { 1.0f, 0.0f, 2.0f });

	vext::core::cpu::operations::unary<vext::core::UnaryOperation::LINEAR>(values.data(), 3, 2.0f, 3.0f);
	expect_vector_near(values, { 5.0f, 3.0f, 7.0f });

	vext::core::cpu::operations::unary<vext::core::UnaryOperation::CLIP>(values.data(), 3, 4.0f, 6.0f);
	expect_vector_near(values, { 5.0f, 4.0f, 6.0f });
}

TEST(CoreCpuUnaryOperations, ComputesNormalizationAcrossWholeBuffer)
{
	std::vector<float> values{ 1.0f, 2.0f, 3.0f };
	const float        denominator = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);

	vext::core::cpu::operations::unary<vext::core::UnaryOperation::SOFTMAX>(values.data(), 3);

	expect_vector_near(values, { std::exp(1.0f) / denominator, std::exp(2.0f) / denominator, std::exp(3.0f) / denominator });
}

TEST(CoreCpuReduction, ReducesEntireBufferWhenNoAxisIsKept)
{
	const std::vector<float> src{ 1, 2, 3, 4 };
	float                    out = 0;

	vext::core::cpu::operations::reduce<vext::core::ReductionOperation::SUM>(
		&out,
		src.data(),
		1,
		4,
		std::vector<std::uint32_t>{ 1 },
		std::vector<std::uint32_t>{ 0 },
		std::vector<std::uint32_t>{ 4 },
		std::vector<std::uint32_t>{ 1 });
	EXPECT_EQ(out, 10.0f);

	vext::core::cpu::operations::reduce<vext::core::ReductionOperation::MEAN>(
		&out,
		src.data(),
		1,
		4,
		std::vector<std::uint32_t>{ 1 },
		std::vector<std::uint32_t>{ 0 },
		std::vector<std::uint32_t>{ 4 },
		std::vector<std::uint32_t>{ 1 });
	EXPECT_EQ(out, 2.5f);
}

TEST(CoreCpuReduction, ReducesSelectedAxisUsingShapeMetadata)
{
	const std::vector<float> src{ 1, 2, 3, 4, 5, 6 };
	std::vector<float>       out(2);

	vext::core::cpu::operations::reduce<vext::core::ReductionOperation::SUM>(
		out.data(),
		src.data(),
		2,
		3,
		std::vector<std::uint32_t>{ 2 },
		std::vector<std::uint32_t>{ 3 },
		std::vector<std::uint32_t>{ 3 },
		std::vector<std::uint32_t>{ 1 });

	expect_vector_near(out, { 6.0f, 15.0f });
}

TEST(CoreCpuLinearAlgebra, ComputesMatrixProduct)
{
	const std::vector<float> lhs{ 1, 2, 3, 4, 5, 6 };
	const std::vector<float> rhs{ 7, 8, 9, 10, 11, 12 };
	std::vector<float>       out(4, 0.0f);

	vext::core::cpu::operations::matmul(out.data(), lhs.data(), rhs.data(), 2, 3, 2);

	expect_vector_near(out, { 58, 64, 139, 154 });
}
