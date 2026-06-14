#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>

#include <gtest/gtest.h>

#include <vext/tensor.hpp>

#include "cpu/allocator.hpp"

class TensorCPUTest
	: public testing::Test
{
protected:
	void
	TearDown() override
	{
		vext::cpu::Allocator::free();
	}
};

void
fill_tensor(
	vext::Tensor&                      tensor,
	const std::initializer_list<float> values)
{
	const vext::Shape&  shape     = tensor.shape();
	const std::uint64_t dim_count = shape.size();

	std::uint64_t i = 0;

	for(const auto value : values)
		{
			if(dim_count == 1)
				{
					tensor[i] = value;
				}
			else if(dim_count == 2)
				{
					tensor[i / shape[1], i % shape[1]] = value;
				}

			++i;
		}
}

void
expect_tensor_eq(
	const vext::Tensor&                tensor,
	const std::initializer_list<float> expected)
{
	const vext::Shape&  shape     = tensor.shape();
	const std::uint64_t dim_count = shape.size();

	std::uint64_t i = 0;

	for(const auto value : expected)
		{
			if(dim_count == 1)
				{
					EXPECT_FLOAT_EQ(tensor[i], value) << "at flat index " << i;
				}
			else if(dim_count == 2)
				{
					EXPECT_FLOAT_EQ((tensor[i / shape[1], i % shape[1]]), value) << "at flat index " << i;
				}

			++i;
		}
}

TEST_F(TensorCPUTest, ConstructorAllocatesZeroInitializedCpuTensor)
{
	const vext::Tensor tensor({ 2, 3 });

	EXPECT_TRUE(tensor.device().is_cpu());
	EXPECT_EQ(tensor.shape(), vext::Shape({ 2, 3 }));
	expect_tensor_eq(tensor, { 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F });
}

TEST_F(TensorCPUTest, IndexingUsesRowMajorStrides)
{
	vext::Tensor tensor({ 2, 3 });

	tensor[0, 0] = 1.0F;
	tensor[0, 1] = 2.0F;
	tensor[0, 2] = 3.0F;
	tensor[1, 0] = 4.0F;
	tensor[1, 1] = 5.0F;
	tensor[1, 2] = 6.0F;

	expect_tensor_eq(tensor, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F });
	EXPECT_FLOAT_EQ((tensor[1, 2]), 6.0F);

	const auto access_invalid_column = [&tensor]()
		{
			(void)tensor[0, 3];
		};

	EXPECT_THROW(access_invalid_column(), std::runtime_error);
	EXPECT_THROW(tensor[6], std::runtime_error);
	EXPECT_THROW(tensor[0], std::runtime_error);
}

TEST_F(TensorCPUTest, CopyConstructorCreatesIndependentStorage)
{
	vext::Tensor source({ 4 });
	fill_tensor(source, { 1.0F, 2.0F, 3.0F, 4.0F });

	vext::Tensor copy(source);
	source[0] = 99.0F;

	EXPECT_EQ(copy.shape(), vext::Shape({ 4 }));
	expect_tensor_eq(copy, { 1.0F, 2.0F, 3.0F, 4.0F });
	expect_tensor_eq(source, { 99.0F, 2.0F, 3.0F, 4.0F });
}

TEST_F(TensorCPUTest, CopyAssignmentReplacesShapeAndCreatesIndependentStorage)
{
	vext::Tensor source({ 4 });
	fill_tensor(source, { 1.0F, 2.0F, 3.0F, 4.0F });

	vext::Tensor copy({ 2 });
	fill_tensor(copy, { -1.0F, -2.0F });

	copy      = source;
	source[1] = 99.0F;

	EXPECT_EQ(copy.shape(), vext::Shape({ 4 }));
	expect_tensor_eq(copy, { 1.0F, 2.0F, 3.0F, 4.0F });
	expect_tensor_eq(source, { 1.0F, 99.0F, 3.0F, 4.0F });
}

TEST_F(TensorCPUTest, MoveConstructionAndAssignmentTransferStorage)
{
	vext::Tensor source({ 3 });
	fill_tensor(source, { 1.0F, 2.0F, 3.0F });

	vext::Tensor moved(std::move(source));

	EXPECT_EQ(moved.shape(), vext::Shape({ 3 }));
	expect_tensor_eq(moved, { 1.0F, 2.0F, 3.0F });

	vext::Tensor assigned({ 1 });
	assigned = std::move(moved);

	EXPECT_EQ(assigned.shape(), vext::Shape({ 3 }));
	expect_tensor_eq(assigned, { 1.0F, 2.0F, 3.0F });
}

TEST_F(TensorCPUTest, ElementwiseOperatorsReturnNewTensorAndPreserveInputs)
{
	vext::Tensor lhs({ 4 });
	vext::Tensor rhs({ 4 });
	fill_tensor(lhs, { 1.0F, -2.0F, 3.5F, 0.0F });
	fill_tensor(rhs, { 4.0F, 5.0F, -1.5F, 2.0F });

	const vext::Tensor sum  = lhs + rhs;
	const vext::Tensor diff = lhs - rhs;
	const vext::Tensor prod = lhs * rhs;

	expect_tensor_eq(sum, { 5.0F, 3.0F, 2.0F, 2.0F });
	expect_tensor_eq(diff, { -3.0F, -7.0F, 5.0F, -2.0F });
	expect_tensor_eq(prod, { 4.0F, -10.0F, -5.25F, 0.0F });
	expect_tensor_eq(lhs, { 1.0F, -2.0F, 3.5F, 0.0F });
	expect_tensor_eq(rhs, { 4.0F, 5.0F, -1.5F, 2.0F });
}

TEST_F(TensorCPUTest, CompoundOperatorsUpdateLeftOperand)
{
	vext::Tensor lhs({ 4 });
	vext::Tensor rhs({ 4 });
	fill_tensor(rhs, { 4.0F, 5.0F, -1.5F, 2.0F });

	fill_tensor(lhs, { 1.0F, -2.0F, 3.5F, 0.0F });
	lhs += rhs;
	expect_tensor_eq(lhs, { 5.0F, 3.0F, 2.0F, 2.0F });

	fill_tensor(lhs, { 1.0F, -2.0F, 3.5F, 0.0F });
	lhs -= rhs;
	expect_tensor_eq(lhs, { -3.0F, -7.0F, 5.0F, -2.0F });

	fill_tensor(lhs, { 1.0F, -2.0F, 3.5F, 0.0F });
	lhs *= rhs;
	expect_tensor_eq(lhs, { 4.0F, -10.0F, -5.25F, 0.0F });
}

TEST_F(TensorCPUTest, ElementwiseOperatorsBroadcastRightOperand)
{
	vext::Tensor lhs({ 2, 3 });
	vext::Tensor rhs({ 3 });
	fill_tensor(lhs, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F });
	fill_tensor(rhs, { 10.0F, 20.0F, 30.0F });

	const vext::Tensor sum  = lhs + rhs;
	const vext::Tensor diff = lhs - rhs;
	const vext::Tensor prod = lhs * rhs;

	EXPECT_EQ(sum.shape(), vext::Shape({ 2, 3 }));
	EXPECT_EQ(diff.shape(), vext::Shape({ 2, 3 }));
	EXPECT_EQ(prod.shape(), vext::Shape({ 2, 3 }));
	expect_tensor_eq(sum, { 11.0F, 22.0F, 33.0F, 14.0F, 25.0F, 36.0F });
	expect_tensor_eq(diff, { -9.0F, -18.0F, -27.0F, -6.0F, -15.0F, -24.0F });
	expect_tensor_eq(prod, { 10.0F, 40.0F, 90.0F, 40.0F, 100.0F, 180.0F });
}

TEST_F(TensorCPUTest, CompoundOperatorsBroadcastRightOperand)
{
	vext::Tensor lhs({ 2, 3 });
	vext::Tensor rhs({ 3 });
	fill_tensor(rhs, { 10.0F, 20.0F, 30.0F });

	fill_tensor(lhs, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F });
	lhs += rhs;
	expect_tensor_eq(lhs, { 11.0F, 22.0F, 33.0F, 14.0F, 25.0F, 36.0F });

	fill_tensor(lhs, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F });
	lhs -= rhs;
	expect_tensor_eq(lhs, { -9.0F, -18.0F, -27.0F, -6.0F, -15.0F, -24.0F });

	fill_tensor(lhs, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F });
	lhs *= rhs;
	expect_tensor_eq(lhs, { 10.0F, 40.0F, 90.0F, 40.0F, 100.0F, 180.0F });
}

TEST_F(TensorCPUTest, MatmulReturnsMatrixProduct)
{
	vext::Tensor lhs({ 2, 3 });
	vext::Tensor rhs({ 3, 2 });
	fill_tensor(lhs, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F });
	fill_tensor(rhs, { 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F });

	const vext::Tensor result = vext::Tensor::matmul(lhs, rhs);

	EXPECT_EQ(result.shape(), vext::Shape({ 2, 2 }));
	expect_tensor_eq(result, { 58.0F, 64.0F, 139.0F, 154.0F });
}

TEST_F(TensorCPUTest, OperatorsRejectIncompatibleShapes)
{
	vext::Tensor lhs({ 2, 3 });
	vext::Tensor rhs({ 4 });
	vext::Tensor matmul_rhs({ 4, 2 });

	EXPECT_THROW(lhs + rhs, std::runtime_error);
	EXPECT_THROW(lhs - rhs, std::runtime_error);
	EXPECT_THROW(lhs * rhs, std::runtime_error);
	EXPECT_THROW(lhs += rhs, std::runtime_error);
	EXPECT_THROW(lhs -= rhs, std::runtime_error);
	EXPECT_THROW(lhs *= rhs, std::runtime_error);
	EXPECT_THROW(vext::Tensor::matmul(lhs, matmul_rhs), std::runtime_error);
}

int
main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
