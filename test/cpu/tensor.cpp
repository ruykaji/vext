#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <numbers>
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

template <typename T1, typename T2>
void
expect_tensor_near(
	const vext::Tensor<T1>& actual,
	const vext::Tensor<T2>& expected,
	const float             tolerance = 1e-5f)
{
	ASSERT_EQ(actual.shape(), expected.shape());

	for(std::uint64_t i = 0; i < actual.shape().length(); ++i)
		{
			EXPECT_NEAR(static_cast<float>(actual.flat_item(i)), static_cast<float>(expected.flat_item(i)), tolerance);
		}
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

TEST(TensorCpu, ConstructsFromShape)
{
	const vext::Tensor<float> tensor(vext::Shape(2, 2));

	expect_shape_eq(tensor.shape(), { 2, 2 });
	expect_tensor_eq(tensor, vext::Tensor<float>({ { 0.0f, 0.0f }, { 0.0f, 0.0f } }));
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

TEST(TensorCpu, MoveConstructorTransfersStorageAndLeavesSourceEmpty)
{
	vext::Tensor<float> source({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });

	const vext::Tensor<float> moved(std::move(source));

	expect_tensor_eq(moved, vext::Tensor<float>({ { 1.0f, 2.0f }, { 3.0f, 4.0f } }));
	expect_shape_eq(source.shape(), {});
}

TEST(TensorCpu, MoveAssignmentTransfersStorageAndLeavesSourceEmpty)
{
	vext::Tensor<float> source({ 3.0f, 4.0f });
	vext::Tensor<float> target({ 1.0f });

	target = std::move(source);

	expect_tensor_eq(target, vext::Tensor<float>({ 3.0f, 4.0f }));
	expect_shape_eq(source.shape(), {});
}

TEST(TensorCpu, FlatItemReadsWritesAndChecksFlatIndices)
{
	vext::Tensor<std::int32_t> tensor({ { 1, 2, 3 }, { 4, 5, 6 } });

	tensor.flat_item(4) = 50;

	EXPECT_EQ(tensor.flat_item(0), 1);
	EXPECT_EQ(tensor.flat_item(4), 50);
	expect_tensor_eq(tensor, vext::Tensor<std::int32_t>({ { 1, 2, 3 }, { 4, 50, 6 } }));
	EXPECT_THROW((void)tensor.flat_item(6), std::invalid_argument);
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

	lhs ^= rhs;
	expect_tensor_eq(lhs, vext::Tensor<std::int32_t>({ 100, 160000, 24300000 }));
}

TEST(TensorCpu, PreluMutatesTensorWithElementwiseSlope)
{
	vext::Tensor<float>       values({ -2.0f, -1.0f, 0.0f, 3.0f });
	const vext::Tensor<float> slopes({ 0.25f, 0.5f, 0.75f, 1.0f });

	values.prelu(slopes);

	expect_tensor_near(values, vext::Tensor<float>({ -0.5f, -0.5f, 0.0f, 3.0f }));
}

TEST(TensorCpu, ParameterlessUnaryOperationsMutateTensor)
{
	vext::Tensor<float> abs_tensor({ -1.0f, 0.0f, 4.0f });
	abs_tensor.abs();
	expect_tensor_near(abs_tensor, vext::Tensor<float>({ 1.0f, 0.0f, 4.0f }));

	vext::Tensor<float> sin_tensor({ 0.0f, static_cast<float>(std::numbers::pi / 2.0) });
	sin_tensor.sin();
	expect_tensor_near(sin_tensor, vext::Tensor<float>({ 0.0f, 1.0f }));

	vext::Tensor<float> cos_tensor({ 0.0f, static_cast<float>(std::numbers::pi) });
	cos_tensor.cos();
	expect_tensor_near(cos_tensor, vext::Tensor<float>({ 1.0f, -1.0f }));

	vext::Tensor<float> exp_tensor({ 0.0f, 1.0f });
	exp_tensor.exp();
	expect_tensor_near(exp_tensor, vext::Tensor<float>({ 1.0f, std::exp(1.0f) }));

	vext::Tensor<float> log_tensor({ 1.0f, std::exp(2.0f) });
	log_tensor.log();
	expect_tensor_near(log_tensor, vext::Tensor<float>({ 0.0f, 2.0f }));

	vext::Tensor<float> sqrt_tensor({ 1.0f, 4.0f, 9.0f });
	sqrt_tensor.sqrt();
	expect_tensor_near(sqrt_tensor, vext::Tensor<float>({ 1.0f, 2.0f, 3.0f }));

	vext::Tensor<float> square_tensor({ -2.0f, 3.0f });
	square_tensor.square();
	expect_tensor_near(square_tensor, vext::Tensor<float>({ 4.0f, 9.0f }));

	vext::Tensor<float> round_tensor({ 1.2f, 1.5f, -1.6f });
	round_tensor.round();
	expect_tensor_near(round_tensor, vext::Tensor<float>({ 1.0f, 2.0f, -2.0f }));
}

TEST(TensorCpu, ActivationUnaryOperationsMutateTensor)
{
	vext::Tensor<float> sigmoid_tensor({ 0.0f, 2.0f });
	sigmoid_tensor.sigmoid();
	expect_tensor_near(sigmoid_tensor, vext::Tensor<float>({ 0.5f, 1.0f / (1.0f + std::exp(-2.0f)) }));

	vext::Tensor<float> soft_relu_tensor({ 0.0f, 2.0f });
	soft_relu_tensor.soft_relu();
	expect_tensor_near(soft_relu_tensor, vext::Tensor<float>({ std::log(2.0f), std::log(1.0f + std::exp(2.0f)) }));

	vext::Tensor<float> relu_tensor({ -2.0f, 0.0f, 3.0f });
	relu_tensor.relu();
	expect_tensor_near(relu_tensor, vext::Tensor<float>({ 0.0f, 0.0f, 3.0f }));

	vext::Tensor<float> leaky_relu_tensor({ -2.0f, 3.0f });
	leaky_relu_tensor.leaky_relu(0.25f);
	expect_tensor_near(leaky_relu_tensor, vext::Tensor<float>({ -0.5f, 3.0f }));

	vext::Tensor<float> elu_tensor({ -1.0f, 2.0f });
	elu_tensor.elu(2.0f);
	expect_tensor_near(elu_tensor, vext::Tensor<float>({ 2.0f * (std::exp(-1.0f) - 1.0f), 2.0f }));

	vext::Tensor<float> swish_tensor({ -1.0f, 2.0f });
	swish_tensor.swish(1.0f);
	expect_tensor_near(swish_tensor, vext::Tensor<float>({ -1.0f / (1.0f + std::exp(1.0f)), 2.0f / (1.0f + std::exp(-2.0f)) }));
}

TEST(TensorCpu, NormalizationUnaryOperationsMutateTensor)
{
	vext::Tensor<float> softmax_tensor({ 1.0f, 2.0f, 3.0f });
	softmax_tensor.softmax();
	const float softmax_sum = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);
	expect_tensor_near(softmax_tensor, vext::Tensor<float>({ std::exp(1.0f) / softmax_sum, std::exp(2.0f) / softmax_sum, std::exp(3.0f) / softmax_sum }));

	vext::Tensor<float> softmin_tensor({ 1.0f, 2.0f, 3.0f });
	softmin_tensor.softmin();
	const float softmin_sum = std::exp(-1.0f) + std::exp(-2.0f) + std::exp(-3.0f);
	expect_tensor_near(softmin_tensor, vext::Tensor<float>({ std::exp(-1.0f) / softmin_sum, std::exp(-2.0f) / softmin_sum, std::exp(-3.0f) / softmin_sum }));

	vext::Tensor<float> log_softmax_tensor({ 1.0f, 2.0f, 3.0f });
	log_softmax_tensor.log_softmax();
	expect_tensor_near(log_softmax_tensor, vext::Tensor<float>({ std::log(std::exp(1.0f) / softmax_sum), std::log(std::exp(2.0f) / softmax_sum), std::log(std::exp(3.0f) / softmax_sum) }));
}

TEST(TensorCpu, ParameterizedUnaryOperationsMutateTensor)
{
	vext::Tensor<float> linear_tensor({ -1.0f, 2.0f });
	linear_tensor.linear(2.0f, 3.0f);
	expect_tensor_near(linear_tensor, vext::Tensor<float>({ 1.0f, 7.0f }));

	vext::Tensor<float> clip_tensor({ -2.0f, 0.5f, 3.0f });
	clip_tensor.clip(-1.0f, 1.0f);
	expect_tensor_near(clip_tensor, vext::Tensor<float>({ -1.0f, 0.5f, 1.0f }));

	vext::Tensor<float> pow_tensor({ 2.0f, 3.0f });
	pow_tensor.pow(2.0f, 3.0f);
	expect_tensor_near(pow_tensor, vext::Tensor<float>({ 16.0f, 54.0f }));
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

TEST(TensorCpu, ReductionRejectsDuplicateAxis)
{
	const vext::Tensor<float> tensor({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });

	EXPECT_THROW((void)tensor.sum(0, 0), std::runtime_error);
}

TEST(TensorCpu, ReductionSupportsNegativeAxis)
{
	const vext::Tensor<float> tensor({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

	expect_tensor_eq(tensor.sum(-1), vext::Tensor<float>({ 6.0f, 15.0f }));
}

TEST(TensorCpu, MatmulComputesMatrixProduct)
{
	const vext::Tensor<float> lhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float> rhs({ { 7.0f, 8.0f }, { 9.0f, 10.0f }, { 11.0f, 12.0f } });

	const auto result = lhs.matmul(rhs);

	expect_tensor_eq(result, vext::Tensor<float>({ { 58.0f, 64.0f }, { 139.0f, 154.0f } }));
}

TEST(TensorCpu, MatmulSupportsBatchedLeftHandTensor)
{
	const vext::Tensor<float> lhs({ { { 1.0f, 2.0f }, { 3.0f, 4.0f } }, { { 5.0f, 6.0f }, { 7.0f, 8.0f } } });
	const vext::Tensor<float> rhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

	const auto result = lhs.matmul(rhs);

	expect_tensor_eq(result, vext::Tensor<float>({ { { 9.0f, 12.0f, 15.0f }, { 19.0f, 26.0f, 33.0f } }, { { 29.0f, 40.0f, 51.0f }, { 39.0f, 54.0f, 69.0f } } }));
}

TEST(TensorCpu, MatmulRejectsIncompatibleShapes)
{
	const vext::Tensor<float> lhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float> rhs({ { 1.0f, 2.0f }, { 3.0f, 4.0f }, { 5.0f, 6.0f }, { 7.0f, 8.0f } });

	EXPECT_THROW((void)lhs.matmul(rhs), std::runtime_error);
}
