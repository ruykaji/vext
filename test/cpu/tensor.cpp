#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <vector>

#include <vext/ops.hpp>

namespace
{

void
expect_shape_eq(
	const std::vector<std::uint32_t>&          shape,
	const std::initializer_list<std::uint32_t> expected)
{
	ASSERT_EQ(shape, std::vector<std::uint32_t>(expected));
	ASSERT_EQ(shape.size(), expected.size());
}

template <typename Tp, typename Up>
void
expect_tensor_values(
	const vext::Tensor<Tp, vext::Backend::CPU>& tensor,
	const std::initializer_list<Up>             expected)
{
	ASSERT_EQ(tensor.length(), expected.size());

	std::uint32_t i = 0;

	for(const auto value : expected)
		{
			EXPECT_EQ(tensor.item(i), static_cast<Tp>(value)) << "at flat index " << i;
			++i;
		}
}

template <typename Tp, typename Up>
void
expect_tensor_near(
	const vext::Tensor<Tp, vext::Backend::CPU>& tensor,
	const std::initializer_list<Up>             expected,
	const float                                 tolerance = 1e-5f)
{
	ASSERT_EQ(tensor.length(), expected.size());

	std::uint32_t i = 0;

	for(const auto value : expected)
		{
			EXPECT_NEAR(static_cast<float>(tensor.item(i)), static_cast<float>(value), tolerance) << "at flat index " << i;
			++i;
		}
}

}

TEST(TensorCpu, ConstructsFromDimensionsWithZeroInitializedStorage)
{
	const vext::Tensor<float> tensor(2, 3);

	expect_shape_eq(tensor.dims(), { 2, 3 });
	expect_tensor_near(tensor, { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f });
}

TEST(TensorCpu, ConstructsFromInitializerList)
{
	const vext::Tensor<std::int32_t> tensor({ { 1, 2, 3 }, { 4, 5, 6 } });

	expect_shape_eq(tensor.dims(), { 2, 3 });
	expect_tensor_values(tensor, { 1, 2, 3, 4, 5, 6 });
}

TEST(TensorCpu, InitializerListRejectsInconsistentShape)
{
	EXPECT_THROW((vext::Tensor<std::int32_t>({ { 1, 2 }, { 3 } })), std::invalid_argument);
}

TEST(TensorCpu, ItemReadsFlatAndMultidimensionalIndices)
{
	const vext::Tensor<float> tensor({ { 1.5f, 2.0f, 3.0f }, { 4.0f, 5.0f, -4.25f } });

	EXPECT_EQ(tensor.item(0), 1.5f);
	EXPECT_EQ(tensor.item(5), -4.25f);
	EXPECT_EQ(tensor.item(0, 1), 2.0f);
	EXPECT_EQ(tensor.item(1, 2), -4.25f);

	EXPECT_THROW((void)tensor.item(0, 0, 0), std::runtime_error);
	EXPECT_THROW((void)tensor.item(2, 0), std::runtime_error);
	EXPECT_THROW((void)tensor.item(0, 3), std::runtime_error);
}

TEST(TensorCpu, MoveConstructorTransfersStorageAndLeavesSourceUsableAsMovedFromObject)
{
	vext::Tensor<float> source({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });

	const vext::Tensor<float> moved(std::move(source));

	expect_shape_eq(moved.dims(), { 2, 2 });
	expect_tensor_near(moved, { 1.0f, 2.0f, 3.0f, 4.0f });
	expect_shape_eq(source.dims(), { vext::core::MIN_LENGTH });
}
