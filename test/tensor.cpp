#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>

#include <gtest/gtest.h>

#include <vext/tensor.hpp>

#include "cpu/allocator.hpp"

class TensorTest
	: public testing::Test
{
protected:
	void
	TearDown() override
	{
		vext::cpu::Allocator::free();
	}

	void
	expect_shape_eq(
		const vext::Shape&                       actual,
		const std::initializer_list<std::size_t> expected)
	{
		ASSERT_EQ(actual.size(), expected.size());

		std::size_t index = 0;

		for(const std::size_t dim : expected)
			{
				EXPECT_EQ(actual.at(index), dim) << "at dimension " << index;
				++index;
			}
	}

	void
	fill_tensor(
		vext::Tensor&                      tensor,
		const std::initializer_list<float> values)
	{
		ASSERT_NE(tensor.data(), nullptr);
		ASSERT_EQ(tensor.shape().length(), values.size());

		std::size_t index = 0;

		for(const float value : values)
			{
				tensor.data()[index] = value;
				++index;
			}
	}

	void
	expect_tensor_eq(
		vext::Tensor&                      tensor,
		const std::initializer_list<float> expected)
	{
		ASSERT_NE(tensor.data(), nullptr);
		ASSERT_EQ(tensor.shape().length(), expected.size());

		std::size_t index = 0;

		for(const float value : expected)
			{
				EXPECT_FLOAT_EQ(tensor.data()[index], value) << "at index " << index;
				++index;
			}
	}

	vext::Tensor
	make_tensor(
		const vext::Shape&                 shape,
		const std::initializer_list<float> values)
	{
		vext::Tensor tensor(shape, vext::Device::cpu());

		fill_tensor(tensor, values);

		return tensor;
	}
};

TEST_F(TensorTest, ConstructorAllocatesZeroInitializedCpuStorage)
{
	vext::Tensor tensor({ 2, 3 }, vext::Device::cpu());

	expect_shape_eq(tensor.shape(), { 2, 3 });
	ASSERT_NE(tensor.data(), nullptr);
	ASSERT_EQ(tensor.shape().length(), 6);
	expect_tensor_eq(tensor, { 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F });
	EXPECT_EQ(tensor.device(), vext::Device::cpu());
}

TEST_F(TensorTest, CopyConstructorCopiesShapeDeviceAndData)
{
	vext::Tensor source({ 2, 2 }, vext::Device::cpu());
	ASSERT_EQ(source.shape().length(), 4);

	fill_tensor(source, { 1.0F, 2.0F, 3.0F, 4.0F });

	vext::Tensor copy(source);

	expect_shape_eq(copy.shape(), { 2, 2 });
	EXPECT_EQ(copy.device(), source.device());
	EXPECT_NE(copy.data(), source.data());
	expect_tensor_eq(copy, { 1.0F, 2.0F, 3.0F, 4.0F });

	source.data()[0] = 10.0F;

	expect_tensor_eq(copy, { 1.0F, 2.0F, 3.0F, 4.0F });
}

TEST_F(TensorTest, MoveConstructorTransfersStorageAndResetsSource)
{
	auto* source = new vext::Tensor({ 2, 2 }, vext::Device::cpu());
	ASSERT_EQ(source->shape().length(), 4);

	fill_tensor(*source, { 1.0F, 2.0F, 3.0F, 4.0F });

	float* data = source->data();

	vext::Tensor moved(std::move(*source));

	expect_shape_eq(moved.shape(), { 2, 2 });
	EXPECT_EQ(moved.data(), data);
	expect_tensor_eq(moved, { 1.0F, 2.0F, 3.0F, 4.0F });
	EXPECT_EQ(source->data(), nullptr);
	EXPECT_EQ(source->shape().size(), 0);

	(void)source;
}

TEST_F(TensorTest, CopyAssignmentCopiesShapeDeviceAndData)
{
	vext::Tensor source({ 2, 2 }, vext::Device::cpu());
	ASSERT_EQ(source.shape().length(), 4);

	fill_tensor(source, { 1.0F, 2.0F, 3.0F, 4.0F });

	vext::Tensor copy({ 1 }, vext::Device::cpu());
	ASSERT_EQ(copy.shape().length(), 1);

	copy = source;

	expect_shape_eq(copy.shape(), { 2, 2 });
	EXPECT_EQ(copy.device(), source.device());
	EXPECT_NE(copy.data(), source.data());
	expect_tensor_eq(copy, { 1.0F, 2.0F, 3.0F, 4.0F });

	source.data()[0] = 10.0F;

	expect_tensor_eq(copy, { 1.0F, 2.0F, 3.0F, 4.0F });
}

TEST_F(TensorTest, MoveAssignmentTransfersStorageAndResetsSource)
{
	auto* source = new vext::Tensor({ 2, 2 }, vext::Device::cpu());
	ASSERT_EQ(source->shape().length(), 4);

	fill_tensor(*source, { 1.0F, 2.0F, 3.0F, 4.0F });

	vext::Tensor moved({ 1 }, vext::Device::cpu());
	ASSERT_EQ(moved.shape().length(), 1);

	float* data = source->data();

	moved = std::move(*source);

	expect_shape_eq(moved.shape(), { 2, 2 });
	EXPECT_EQ(moved.data(), data);
	expect_tensor_eq(moved, { 1.0F, 2.0F, 3.0F, 4.0F });
	EXPECT_EQ(source->data(), nullptr);
	EXPECT_EQ(source->shape().size(), 0);

	(void)source;
}

TEST_F(TensorTest, AdditionCreatesElementwiseSum)
{
	vext::Tensor lhs({ 2, 2 }, vext::Device::cpu());
	ASSERT_EQ(lhs.shape().length(), 4);

	fill_tensor(lhs, { 1.0F, -2.0F, 3.5F, 0.0F });

	vext::Tensor rhs    = make_tensor({ 2, 2 }, { 4.0F, 5.0F, -1.5F, 2.0F });
	vext::Tensor result = lhs + rhs;

	expect_shape_eq(result.shape(), { 2, 2 });
	expect_tensor_eq(result, { 5.0F, 3.0F, 2.0F, 2.0F });
	expect_tensor_eq(lhs, { 1.0F, -2.0F, 3.5F, 0.0F });
	expect_tensor_eq(rhs, { 4.0F, 5.0F, -1.5F, 2.0F });
}

TEST_F(TensorTest, SubtractionCreatesElementwiseDifference)
{
	vext::Tensor lhs({ 2, 2 }, vext::Device::cpu());
	ASSERT_EQ(lhs.shape().length(), 4);

	fill_tensor(lhs, { 1.0F, -2.0F, 3.5F, 0.0F });

	vext::Tensor rhs    = make_tensor({ 2, 2 }, { 4.0F, 5.0F, -1.5F, 2.0F });
	vext::Tensor result = lhs - rhs;

	expect_shape_eq(result.shape(), { 2, 2 });
	expect_tensor_eq(result, { -3.0F, -7.0F, 5.0F, -2.0F });
	expect_tensor_eq(lhs, { 1.0F, -2.0F, 3.5F, 0.0F });
	expect_tensor_eq(rhs, { 4.0F, 5.0F, -1.5F, 2.0F });
}

TEST_F(TensorTest, MultiplicationCreatesMatrixProduct)
{
	vext::Tensor lhs({ 2, 3 }, vext::Device::cpu());
	ASSERT_EQ(lhs.shape().length(), 6);

	fill_tensor(lhs, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F });

	vext::Tensor rhs    = make_tensor({ 3, 2 }, { 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F });
	vext::Tensor result = lhs * rhs;

	expect_shape_eq(result.shape(), { 2, 2 });
	expect_tensor_eq(result, { 58.0F, 64.0F, 139.0F, 154.0F });
}

TEST_F(TensorTest, InPlaceOperationsUpdateLeftOperand)
{
	vext::Tensor lhs({ 2, 2 }, vext::Device::cpu());
	ASSERT_EQ(lhs.shape().length(), 4);

	fill_tensor(lhs, { 1.0F, -2.0F, 3.5F, 2.0F });

	vext::Tensor rhs = make_tensor({ 2, 2 }, { 4.0F, 5.0F, -1.5F, 3.0F });

	lhs += rhs;
	expect_tensor_eq(lhs, { 5.0F, 3.0F, 2.0F, 5.0F });

	lhs -= rhs;
	expect_tensor_eq(lhs, { 1.0F, -2.0F, 3.5F, 2.0F });

	lhs *= rhs;
	expect_tensor_eq(lhs, { 4.0F, -10.0F, -5.25F, 6.0F });
	expect_tensor_eq(rhs, { 4.0F, 5.0F, -1.5F, 3.0F });
}

TEST_F(TensorTest, OperationsRejectDifferentShapes)
{
	vext::Tensor lhs({ 2, 2 }, vext::Device::cpu());
	ASSERT_EQ(lhs.shape().length(), 4);

	vext::Tensor rhs({ 4 }, vext::Device::cpu());
	ASSERT_EQ(rhs.shape().length(), 4);

	EXPECT_THROW(lhs + rhs, std::runtime_error);
	EXPECT_THROW(lhs - rhs, std::runtime_error);
	EXPECT_THROW(lhs += rhs, std::runtime_error);
	EXPECT_THROW(lhs -= rhs, std::runtime_error);
	EXPECT_THROW(lhs *= rhs, std::runtime_error);
}

TEST_F(TensorTest, OperationsRejectDifferentDevices)
{
	vext::Tensor lhs({ 2, 2 }, vext::Device::cpu());
	ASSERT_EQ(lhs.shape().length(), 4);

	vext::Tensor rhs({ 2, 2 }, vext::Device(vext::type::Backend::CPU, 1));
	ASSERT_EQ(rhs.shape().length(), 4);

	EXPECT_THROW(lhs + rhs, std::runtime_error);
	EXPECT_THROW(lhs - rhs, std::runtime_error);
	EXPECT_THROW(lhs * rhs, std::runtime_error);
	EXPECT_THROW(lhs += rhs, std::runtime_error);
	EXPECT_THROW(lhs -= rhs, std::runtime_error);
	EXPECT_THROW(lhs *= rhs, std::runtime_error);
}

TEST_F(TensorTest, MatrixMultiplicationRejectsMismatchedSharedDimension)
{
	vext::Tensor lhs({ 2, 3 }, vext::Device::cpu());
	ASSERT_EQ(lhs.shape().length(), 6);

	vext::Tensor rhs({ 2, 2 }, vext::Device::cpu());
	ASSERT_EQ(rhs.shape().length(), 4);

	EXPECT_THROW(lhs * rhs, std::runtime_error);
}

int
main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
