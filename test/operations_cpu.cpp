#include <array>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "cpu/operations.hpp"

template <std::uint64_t N>
void
expect_elements_eq(
	const std::array<float, N>& actual,
	const std::array<float, N>& expected)
{
	for(std::uint64_t i = 0; i < N; ++i)
		{
			EXPECT_FLOAT_EQ(actual[i], expected[i]) << "at index " << i;
		}
}

TEST(OperationsCPUTest, SumWritesToDestination)
{
	std::array<float, 4> dst = { -1.0F, -1.0F, -1.0F, -1.0F };
	std::array<float, 4> a   = { 1.0F, -2.0F, 3.5F, 0.0F };
	std::array<float, 4> b   = { 4.0F, 5.0F, -1.5F, 2.0F };

	vext::cpu::Operations::sum(dst.data(), a.data(), b.data(), dst.size());

	expect_elements_eq(dst, { 5.0F, 3.0F, 2.0F, 2.0F });
	expect_elements_eq(a, { 1.0F, -2.0F, 3.5F, 0.0F });
	expect_elements_eq(b, { 4.0F, 5.0F, -1.5F, 2.0F });
}

TEST(OperationsCPUTest, SumUpdatesLeftOperand)
{
	std::array<float, 4> a = { 1.0F, -2.0F, 3.5F, 0.0F };
	std::array<float, 4> b = { 4.0F, 5.0F, -1.5F, 2.0F };

	vext::cpu::Operations::sum(a.data(), a.data(), b.data(), a.size());

	expect_elements_eq(a, { 5.0F, 3.0F, 2.0F, 2.0F });
	expect_elements_eq(b, { 4.0F, 5.0F, -1.5F, 2.0F });
}

TEST(OperationsCPUTest, DiffWritesToDestination)
{
	std::array<float, 4> dst = { -1.0F, -1.0F, -1.0F, -1.0F };
	std::array<float, 4> a   = { 1.0F, -2.0F, 3.5F, 0.0F };
	std::array<float, 4> b   = { 4.0F, 5.0F, -1.5F, 2.0F };

	vext::cpu::Operations::diff(dst.data(), a.data(), b.data(), dst.size());

	expect_elements_eq(dst, { -3.0F, -7.0F, 5.0F, -2.0F });
	expect_elements_eq(a, { 1.0F, -2.0F, 3.5F, 0.0F });
	expect_elements_eq(b, { 4.0F, 5.0F, -1.5F, 2.0F });
}

TEST(OperationsCPUTest, DiffUpdatesLeftOperand)
{
	std::array<float, 4> a = { 1.0F, -2.0F, 3.5F, 0.0F };
	std::array<float, 4> b = { 4.0F, 5.0F, -1.5F, 2.0F };

	vext::cpu::Operations::diff(a.data(), a.data(), b.data(), a.size());

	expect_elements_eq(a, { -3.0F, -7.0F, 5.0F, -2.0F });
	expect_elements_eq(b, { 4.0F, 5.0F, -1.5F, 2.0F });
}

TEST(OperationsCPUTest, MulWritesElementwiseProductToDestination)
{
	std::array<float, 4> dst = { -1.0F, -1.0F, -1.0F, -1.0F };
	std::array<float, 4> a   = { 1.0F, -2.0F, 3.5F, 0.0F };
	std::array<float, 4> b   = { 4.0F, 5.0F, -1.5F, 2.0F };

	vext::cpu::Operations::mul(dst.data(), a.data(), b.data(), dst.size());

	expect_elements_eq(dst, { 4.0F, -10.0F, -5.25F, 0.0F });
	expect_elements_eq(a, { 1.0F, -2.0F, 3.5F, 0.0F });
	expect_elements_eq(b, { 4.0F, 5.0F, -1.5F, 2.0F });
}

TEST(OperationsCPUTest, MulUpdatesLeftOperandElementwise)
{
	std::array<float, 4> a = { 1.0F, -2.0F, 3.5F, 0.0F };
	std::array<float, 4> b = { 4.0F, 5.0F, -1.5F, 2.0F };

	vext::cpu::Operations::mul(a.data(), a.data(), b.data(), a.size());

	expect_elements_eq(a, { 4.0F, -10.0F, -5.25F, 0.0F });
	expect_elements_eq(b, { 4.0F, 5.0F, -1.5F, 2.0F });
}

TEST(OperationsCPUTest, MulWritesMatrixProductToDestination)
{
	std::array<float, 4> dst = { 0.0F, 0.0F, 0.0F, 0.0F };
	std::array<float, 6> a   = {
		1.0F, 2.0F, 3.0F,
		4.0F, 5.0F, 6.0F
	};
	std::array<float, 6> b = {
		7.0F, 8.0F,
		9.0F, 10.0F,
		11.0F, 12.0F
	};

	vext::cpu::Operations::mul(dst.data(), a.data(), b.data(), 2, 3, 2);

	expect_elements_eq(dst, { 58.0F, 64.0F, 139.0F, 154.0F });
	expect_elements_eq(a, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F });
	expect_elements_eq(b, { 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F });
}

TEST(OperationsCPUTest, SumWritesBroadcastedValuesToDestination)
{
	std::array<float, 6> dst = { -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F };
	std::array<float, 6> a   = {
		1.0F, 2.0F, 3.0F,
		4.0F, 5.0F, 6.0F
	};
	std::array<float, 3>             b       = { 10.0F, 20.0F, 30.0F };
	const std::vector<std::uint64_t> dims    = { 2, 3 };
	const std::vector<std::uint64_t> strides = { 0, 1 };

	vext::cpu::Operations::sum(dst.data(), a.data(), b.data(), dst.size(), dims, strides);

	expect_elements_eq(dst, { 11.0F, 22.0F, 33.0F, 14.0F, 25.0F, 36.0F });
	expect_elements_eq(a, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F });
	expect_elements_eq(b, { 10.0F, 20.0F, 30.0F });
}

TEST(OperationsCPUTest, DiffWritesBroadcastedValuesToDestination)
{
	std::array<float, 6> dst = { -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F };
	std::array<float, 6> a   = {
		1.0F, 2.0F, 3.0F,
		4.0F, 5.0F, 6.0F
	};
	std::array<float, 3>             b       = { 10.0F, 20.0F, 30.0F };
	const std::vector<std::uint64_t> dims    = { 2, 3 };
	const std::vector<std::uint64_t> strides = { 0, 1 };

	vext::cpu::Operations::diff(dst.data(), a.data(), b.data(), dst.size(), dims, strides);

	expect_elements_eq(dst, { -9.0F, -18.0F, -27.0F, -6.0F, -15.0F, -24.0F });
	expect_elements_eq(a, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F });
	expect_elements_eq(b, { 10.0F, 20.0F, 30.0F });
}

TEST(OperationsCPUTest, MulWritesBroadcastedValuesToDestination)
{
	std::array<float, 6> dst = { -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F };
	std::array<float, 6> a   = {
		1.0F, 2.0F, 3.0F,
		4.0F, 5.0F, 6.0F
	};
	std::array<float, 3>             b       = { 10.0F, 20.0F, 30.0F };
	const std::vector<std::uint64_t> dims    = { 2, 3 };
	const std::vector<std::uint64_t> strides = { 0, 1 };

	vext::cpu::Operations::mul(dst.data(), a.data(), b.data(), dst.size(), dims, strides);

	expect_elements_eq(dst, { 10.0F, 40.0F, 90.0F, 40.0F, 100.0F, 180.0F });
	expect_elements_eq(a, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F });
	expect_elements_eq(b, { 10.0F, 20.0F, 30.0F });
}

TEST(OperationsCPUTest, ZeroLengthOperationsLeaveDestinationUntouched)
{
	std::array<float, 1> dst = { 42.0F };
	std::array<float, 1> a   = { 1.0F };
	std::array<float, 1> b   = { 2.0F };

	vext::cpu::Operations::sum(dst.data(), a.data(), b.data(), 0);
	vext::cpu::Operations::diff(dst.data(), a.data(), b.data(), 0);
	vext::cpu::Operations::mul(dst.data(), a.data(), b.data(), 0);
	vext::cpu::Operations::sum(a.data(), a.data(), b.data(), 0);
	vext::cpu::Operations::diff(a.data(), a.data(), b.data(), 0);
	vext::cpu::Operations::mul(a.data(), a.data(), b.data(), 0);

	EXPECT_FLOAT_EQ(dst[0], 42.0F);
	EXPECT_FLOAT_EQ(a[0], 1.0F);
	EXPECT_FLOAT_EQ(b[0], 2.0F);
}

int
main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
