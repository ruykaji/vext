#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <type_traits>
#include <vector>

#include <vext/tensor.hpp>

namespace
{

void
expect_shape_eq(
	const vext::Shape&                         shape,
	const std::initializer_list<std::uint64_t> expected)
{
	ASSERT_EQ(shape.dims(), std::vector<std::uint64_t>(expected));
	ASSERT_EQ(shape.size(), expected.size());
}

template <typename T1, typename T2>
void
expect_tensor_eq(
	const vext::Tensor<T1>& actual,
	const vext::Tensor<T2>& expected)
{
	ASSERT_EQ(actual.shape(), expected.shape());
	ASSERT_EQ((actual == expected).min().item(0), 1);
}

template <typename T>
void
expect_scalar_eq(
	const vext::Tensor<T>& tensor,
	const T                expected)
{
	expect_shape_eq(tensor.shape(), { 1 });
	EXPECT_EQ(tensor.item(0), expected);
}

template <typename T>
void
expect_scalar_near(
	const vext::Tensor<T>& tensor,
	const T                expected,
	const T                tolerance = static_cast<T>(1e-5))
{
	expect_shape_eq(tensor.shape(), { 1 });
	EXPECT_NEAR(tensor.item(0), expected, tolerance);
}

}

TEST(TensorCpu, ConstructsFromDimensionsWithZeroInitializedStorage)
{
	const vext::Tensor<float> tensor(2, 3);
	const vext::Tensor<float> expected({ { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } });

	expect_shape_eq(tensor.shape(), { 2, 3 });
	expect_tensor_eq(tensor, expected);
}

TEST(TensorCpu, ConstructsFromInitializerList)
{
	const vext::Tensor<std::int32_t> tensor({ { 1, 2, 3 }, { 4, 5, 6 } });
	const vext::Tensor<std::int32_t> expected({ { 1, 2, 3 }, { 4, 5, 6 } });

	expect_shape_eq(tensor.shape(), { 2, 3 });
	expect_tensor_eq(tensor, expected);
}

TEST(TensorCpu, InitializerListRejectsInconsistentShape)
{
	EXPECT_THROW((vext::Tensor<std::int32_t>({ { 1, 2 }, { 3 } })), std::invalid_argument);
}

TEST(TensorCpu, ItemReadsWritesAndChecksIndices)
{
	vext::Tensor<float> tensor(2, 3);

	tensor.item(0, 0) = 1.5f;
	tensor.item(1, 2) = -4.25f;

	const vext::Tensor<float> expected({ { 1.5f, 0.0f, 0.0f }, { 0.0f, 0.0f, -4.25f } });

	expect_tensor_eq(tensor, expected);
	EXPECT_THROW((void)tensor.item(0), std::runtime_error);
	EXPECT_THROW((void)tensor.item(2, 0), std::runtime_error);
	EXPECT_THROW((void)tensor.item(0, 3), std::runtime_error);
}

TEST(TensorCpu, CopyConstructorCreatesIndependentStorage)
{
	vext::Tensor<float>       original({ 1.0f, 2.0f, 3.0f });
	const vext::Tensor<float> expected_copy({ 1.0f, 2.0f, 3.0f });

	vext::Tensor<float> copy(original);
	original.item(1) = 42.0f;

	expect_tensor_eq(copy, expected_copy);
	expect_tensor_eq(original, vext::Tensor<float>({ 1.0f, 42.0f, 3.0f }));
}

TEST(TensorCpu, CopyAssignmentCreatesIndependentStorage)
{
	vext::Tensor<float> source({ 7.0f, 9.0f });
	vext::Tensor<float> target(1);

	target         = source;
	source.item(0) = 100.0f;

	expect_tensor_eq(target, vext::Tensor<float>({ 7.0f, 9.0f }));
	expect_tensor_eq(source, vext::Tensor<float>({ 100.0f, 9.0f }));
}

TEST(TensorCpu, LogicalComparisonsProduceMaskTensors)
{
	const vext::Tensor<std::int32_t> lhs({ 1, 2, 3, 4 });
	const vext::Tensor<std::int32_t> rhs({ 1, 0, 3, 5 });

	expect_tensor_eq(lhs == rhs, vext::Tensor<std::uint8_t>({ 1, 0, 1, 0 }));
	expect_tensor_eq(lhs != rhs, vext::Tensor<std::uint8_t>({ 0, 1, 0, 1 }));
	expect_tensor_eq(lhs < rhs, vext::Tensor<std::uint8_t>({ 0, 0, 0, 1 }));
	expect_tensor_eq(lhs <= rhs, vext::Tensor<std::uint8_t>({ 1, 0, 1, 1 }));
	expect_tensor_eq(lhs > rhs, vext::Tensor<std::uint8_t>({ 0, 1, 0, 0 }));
	expect_tensor_eq(lhs >= rhs, vext::Tensor<std::uint8_t>({ 1, 1, 1, 0 }));
}

TEST(TensorCpu, EqualityCanBeReducedToAllElementsMatch)
{
	const vext::Tensor<float> lhs({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });
	const vext::Tensor<float> same({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });
	const vext::Tensor<float> different({ { 1.0f, 2.0f }, { 3.0f, 5.0f } });

	EXPECT_EQ((lhs == same).min().item(0), 1);
	EXPECT_EQ((lhs == different).min().item(0), 0);
}

TEST(TensorCpu, LogicalComparisonsRejectIncompatibleShapes)
{
	const vext::Tensor<float> lhs({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });
	const vext::Tensor<float> rhs({ 1.0f, 2.0f });

	EXPECT_THROW((void)(lhs == rhs), std::runtime_error);
}

TEST(TensorCpu, ElementwiseArithmeticSupportsSameShape)
{
	const vext::Tensor<std::int32_t> lhs({ 8, 12, 20 });
	const vext::Tensor<std::int32_t> rhs({ 2, 3, 4 });

	const auto sum        = lhs + rhs;
	const auto difference = lhs - rhs;
	const auto product    = lhs * rhs;
	const auto quotient   = lhs / rhs;
	const auto power      = rhs ^ rhs;

	static_assert(std::is_same_v<decltype(sum), const vext::Tensor<std::int32_t>>);

	expect_tensor_eq(sum, vext::Tensor<std::int32_t>({ 10, 15, 24 }));
	expect_tensor_eq(difference, vext::Tensor<std::int32_t>({ 6, 9, 16 }));
	expect_tensor_eq(product, vext::Tensor<std::int32_t>({ 16, 36, 80 }));
	expect_tensor_eq(quotient, vext::Tensor<std::int32_t>({ 4, 4, 5 }));
	expect_tensor_eq(power, vext::Tensor<std::int32_t>({ 4, 27, 256 }));
}

TEST(TensorCpu, ElementwiseArithmeticUsesCommonType)
{
	const vext::Tensor<std::int32_t> lhs({ 1, 2 });
	const vext::Tensor<float>        rhs({ 0.5f, 1.25f });

	const auto result = lhs + rhs;

	static_assert(std::is_same_v<decltype(result), const vext::Tensor<float>>);
	expect_tensor_eq(result, vext::Tensor<float>({ 1.5f, 3.25f }));
}

TEST(TensorCpu, ElementwiseArithmeticBroadcastsRightHandTensor)
{
	const vext::Tensor<float> matrix({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float> bias({ 10.0f, 20.0f, 30.0f });

	const auto result = matrix + bias;

	expect_tensor_eq(result, vext::Tensor<float>({ { 11.0f, 22.0f, 33.0f }, { 14.0f, 25.0f, 36.0f } }));
}

TEST(TensorCpu, ElementwiseArithmeticRejectsIncompatibleShapes)
{
	const vext::Tensor<float> lhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float> rhs({ 1.0f, 2.0f, 3.0f, 4.0f });

	EXPECT_THROW((void)(lhs + rhs), std::runtime_error);
}

TEST(TensorCpu, InPlaceArithmeticMutatesLeftHandSide)
{
	vext::Tensor<std::int32_t>       lhs({ 10, 20, 30 });
	const vext::Tensor<std::int32_t> rhs({ 2, 4, 5 });

	lhs += rhs;
	expect_tensor_eq(lhs, vext::Tensor<std::int32_t>({ 12, 24, 35 }));

	lhs -= rhs;
	expect_tensor_eq(lhs, vext::Tensor<std::int32_t>({ 10, 20, 30 }));

	lhs *= rhs;
	expect_tensor_eq(lhs, vext::Tensor<std::int32_t>({ 20, 80, 150 }));

	lhs /= rhs;
	expect_tensor_eq(lhs, vext::Tensor<std::int32_t>({ 10, 20, 30 }));
}

TEST(TensorCpu, UnaryOperationsMutateTensor)
{
	vext::Tensor<float> tensor({ -1.0f, 0.0f, 4.0f });

	tensor.abs();
	expect_tensor_eq(tensor, vext::Tensor<float>({ 1.0f, 0.0f, 4.0f }));

	tensor.sqrt();
	expect_tensor_eq(tensor, vext::Tensor<float>({ 1.0f, 0.0f, 2.0f }));

	tensor.exp();
	expect_tensor_eq(tensor, vext::Tensor<float>({ std::exp(1.0f), 1.0f, std::exp(2.0f) }));
}

TEST(TensorCpu, UnaryMinusNegatesInPlace)
{
	vext::Tensor<std::int32_t> tensor({ 1, -2, 3 });

	-tensor;

	expect_tensor_eq(tensor, vext::Tensor<std::int32_t>({ -1, 2, -3 }));
}

TEST(TensorCpu, ReductionsWithoutAxisReturnScalarTensor)
{
	vext::Tensor<float> tensor({ 1.0f, 2.0f, 3.0f, 4.0f });

	expect_scalar_eq(tensor.sum(), 10.0f);
	expect_scalar_eq(tensor.prod(), 24.0f);
	expect_scalar_eq(tensor.min(), 1.0f);
	expect_scalar_eq(tensor.max(), 4.0f);
	expect_scalar_eq(tensor.mean(), 2.5f);
	expect_scalar_eq(tensor.var(), 1.25f);
	expect_scalar_near(tensor.std(), std::sqrt(1.25f));
}

TEST(TensorCpu, ReductionsAlongAxis)
{
	vext::Tensor<float> tensor({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

	expect_tensor_eq(tensor.sum(1), vext::Tensor<float>({ 6.0f, 15.0f }));
	expect_tensor_eq(tensor.sum(0), vext::Tensor<float>({ 5.0f, 7.0f, 9.0f }));
	expect_tensor_eq(tensor.mean(1), vext::Tensor<float>({ 2.0f, 5.0f }));
	expect_tensor_eq(tensor.max(0), vext::Tensor<float>({ 4.0f, 5.0f, 6.0f }));
	expect_tensor_eq(tensor.min(1), vext::Tensor<float>({ 1.0f, 4.0f }));
	expect_tensor_eq(tensor.prod(1), vext::Tensor<float>({ 6.0f, 120.0f }));
}

TEST(TensorCpu, StatisticalReductionsAlongAxis)
{
	vext::Tensor<float> tensor({ { 1.0f, 3.0f }, { 5.0f, 7.0f } });

	expect_tensor_eq(tensor.var(1), vext::Tensor<float>({ 1.0f, 1.0f }));
	expect_tensor_eq(tensor.std(1), vext::Tensor<float>({ 1.0f, 1.0f }));
}

TEST(TensorCpu, ReductionRejectsMissingAxis)
{
	const vext::Tensor<float> tensor({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

	EXPECT_THROW((void)tensor.sum(2), std::runtime_error);
}

TEST(TensorCpu, MatmulComputesMatrixProduct)
{
	const vext::Tensor<float> lhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float> rhs({ { 7.0f, 8.0f }, { 9.0f, 10.0f }, { 11.0f, 12.0f } });

	const auto result = lhs.matmul(rhs);

	expect_tensor_eq(result, vext::Tensor<float>({ { 58.0f, 64.0f }, { 139.0f, 154.0f } }));
}

TEST(TensorCpu, MatmulRejectsIncompatibleShapes)
{
	const vext::Tensor<float> lhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float> rhs({ { 1.0f, 2.0f }, { 3.0f, 4.0f }, { 5.0f, 6.0f }, { 7.0f, 8.0f } });

	EXPECT_THROW((void)lhs.matmul(rhs), std::runtime_error);
}
