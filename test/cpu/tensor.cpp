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
	const std::initializer_list<std::uint32_t> expected)
{
	ASSERT_EQ(shape.dims(), std::vector<std::uint32_t>(expected));
	ASSERT_EQ(shape.size(), expected.size());
}

template <typename Tp, typename Up>
void
expect_tensor_values(
	const vext::Tensor<Tp, vext::Backend::CPU>& tensor,
	const std::initializer_list<Up>             expected)
{
	ASSERT_EQ(tensor.shape().length(), expected.size());

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
	ASSERT_EQ(tensor.shape().length(), expected.size());

	std::uint32_t i = 0;

	for(const auto value : expected)
		{
			EXPECT_NEAR(static_cast<float>(tensor.item(i)), static_cast<float>(value), tolerance) << "at flat index " << i;
			++i;
		}
}

template <typename Tp>
void
expect_scalar_eq(
	const vext::Tensor<Tp, vext::Backend::CPU>& tensor,
	const Tp                                    expected)
{
	expect_shape_eq(tensor.shape(), { 1 });
	EXPECT_EQ(tensor.item(0), expected);
}

template <typename Tp>
void
expect_scalar_near(
	const vext::Tensor<Tp, vext::Backend::CPU>& tensor,
	const Tp                                    expected,
	const Tp                                    tolerance = static_cast<Tp>(1e-5))
{
	expect_shape_eq(tensor.shape(), { 1 });
	EXPECT_NEAR(tensor.item(0), expected, tolerance);
}

}

TEST(TensorCpu, ConstructsFromDimensionsWithZeroInitializedStorage)
{
	const vext::Tensor<float> tensor(2, 3);

	expect_shape_eq(tensor.shape(), { 2, 3 });
	expect_tensor_near(tensor, { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f });
}

TEST(TensorCpu, ConstructsFromInitializerList)
{
	const vext::Tensor<std::int32_t> tensor({ { 1, 2, 3 }, { 4, 5, 6 } });

	expect_shape_eq(tensor.shape(), { 2, 3 });
	expect_tensor_values(tensor, { 1, 2, 3, 4, 5, 6 });
}

TEST(TensorCpu, ConstructsFromShape)
{
	const vext::Tensor<float> tensor(vext::Shape(2, 2));

	expect_shape_eq(tensor.shape(), { 2, 2 });
	expect_tensor_near(tensor, { 0.0f, 0.0f, 0.0f, 0.0f });
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

TEST(TensorCpu, CopyConstructorCreatesIndependentStorage)
{
	vext::Tensor<float> original({ 1.0f, 2.0f, 3.0f });
	vext::Tensor<float> copy(original);

	original += vext::Tensor<float>({ 10.0f, 10.0f, 10.0f });

	expect_tensor_near(copy, { 1.0f, 2.0f, 3.0f });
	expect_tensor_near(original, { 11.0f, 12.0f, 13.0f });
}

TEST(TensorCpu, CopyAssignmentCreatesIndependentStorage)
{
	vext::Tensor<float> source({ 7.0f, 9.0f });
	vext::Tensor<float> target(1);

	target = source;
	source *= vext::Tensor<float>({ 2.0f, 3.0f });

	expect_tensor_near(target, { 7.0f, 9.0f });
	expect_tensor_near(source, { 14.0f, 27.0f });
}

TEST(TensorCpu, MoveConstructorTransfersStorageAndLeavesSourceUsableAsMovedFromObject)
{
	vext::Tensor<float> source({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });

	const vext::Tensor<float> moved(std::move(source));

	expect_shape_eq(moved.shape(), { 2, 2 });
	expect_tensor_near(moved, { 1.0f, 2.0f, 3.0f, 4.0f });
	expect_shape_eq(source.shape(), { vext::core::MIN_LENGTH });
}

TEST(TensorCpu, LogicalComparisonsProduceMaskTensors)
{
	const vext::Tensor<std::int32_t> lhs({ 1, 2, 3, 4 });
	const vext::Tensor<std::int32_t> rhs({ 1, 0, 3, 5 });

	expect_tensor_values(lhs == rhs, { 1, 0, 1, 0 });
	expect_tensor_values(lhs != rhs, { 0, 1, 0, 1 });
	expect_tensor_values(lhs < rhs, { 0, 0, 0, 1 });
	expect_tensor_values(lhs <= rhs, { 1, 0, 1, 1 });
	expect_tensor_values(lhs > rhs, { 0, 1, 0, 0 });
	expect_tensor_values(lhs >= rhs, { 1, 1, 1, 0 });
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

	const vext::Tensor<std::int32_t> sum        = lhs + rhs;
	const vext::Tensor<std::int32_t> difference = lhs - rhs;
	const vext::Tensor<std::int32_t> product    = lhs * rhs;
	const vext::Tensor<std::int32_t> quotient   = lhs / rhs;
	const vext::Tensor<std::int32_t> power      = rhs ^ rhs;

	static_assert(std::is_same_v<decltype(sum), const vext::Tensor<std::int32_t>>);

	expect_tensor_values(sum, { 10, 15, 24 });
	expect_tensor_values(difference, { 6, 9, 16 });
	expect_tensor_values(product, { 16, 36, 80 });
	expect_tensor_values(quotient, { 4, 4, 5 });
	expect_tensor_values(power, { 4, 27, 256 });
}

TEST(TensorCpu, ElementwiseArithmeticUsesCommonType)
{
	const vext::Tensor<std::int32_t> lhs({ 1, 2 });
	const vext::Tensor<float>        rhs({ 0.5f, 1.25f });

	const vext::Tensor<float> result = lhs + rhs;

	static_assert(std::is_same_v<decltype(result), const vext::Tensor<float>>);
	expect_tensor_near(result, { 1.5f, 3.25f });
}

TEST(TensorCpu, ElementwiseArithmeticBroadcastsRightHandTensor)
{
	const vext::Tensor<float> matrix({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float> bias({ 10.0f, 20.0f, 30.0f });

	const vext::Tensor<float> result = matrix + bias;

	expect_shape_eq(result.shape(), { 2, 3 });
	expect_tensor_near(result, { 11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f });
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
	expect_tensor_values(lhs, { 12, 24, 35 });

	lhs -= rhs;
	expect_tensor_values(lhs, { 10, 20, 30 });

	lhs *= rhs;
	expect_tensor_values(lhs, { 20, 80, 150 });

	lhs /= rhs;
	expect_tensor_values(lhs, { 10, 20, 30 });

	lhs ^= rhs;
	expect_tensor_values(lhs, { 100, 160000, 24300000 });
}

TEST(TensorCpu, PreluMutatesTensorWithElementwiseSlope)
{
	vext::Tensor<float>       values({ -2.0f, -1.0f, 0.0f, 3.0f });
	const vext::Tensor<float> slopes({ 0.25f, 0.5f, 0.75f, 1.0f });

	values.prelu(slopes);

	expect_tensor_near(values, { -0.5f, -0.5f, 0.0f, 3.0f });
}

TEST(TensorCpu, ParameterlessUnaryOperationsMutateTensor)
{
	vext::Tensor<float> abs_tensor({ -1.0f, 0.0f, 4.0f });
	abs_tensor.abs();
	expect_tensor_near(abs_tensor, { 1.0f, 0.0f, 4.0f });

	vext::Tensor<float> sin_tensor({ 0.0f, static_cast<float>(std::numbers::pi / 2.0) });
	sin_tensor.sin();
	expect_tensor_near(sin_tensor, { 0.0f, 1.0f });

	vext::Tensor<float> cos_tensor({ 0.0f, static_cast<float>(std::numbers::pi) });
	cos_tensor.cos();
	expect_tensor_near(cos_tensor, { 1.0f, -1.0f });

	vext::Tensor<float> exp_tensor({ 0.0f, 1.0f });
	exp_tensor.exp();
	expect_tensor_near(exp_tensor, { 1.0f, std::exp(1.0f) });

	vext::Tensor<float> log_tensor({ 1.0f, std::exp(2.0f) });
	log_tensor.log();
	expect_tensor_near(log_tensor, { 0.0f, 2.0f });

	vext::Tensor<float> sqrt_tensor({ 1.0f, 4.0f, 9.0f });
	sqrt_tensor.sqrt();
	expect_tensor_near(sqrt_tensor, { 1.0f, 2.0f, 3.0f });

	vext::Tensor<float> square_tensor({ -2.0f, 3.0f });
	square_tensor.square();
	expect_tensor_near(square_tensor, { 4.0f, 9.0f });

	vext::Tensor<float> round_tensor({ 1.2f, 1.5f, -1.6f });
	round_tensor.round();
	expect_tensor_near(round_tensor, { 1.0f, 2.0f, -2.0f });
}

TEST(TensorCpu, ActivationUnaryOperationsMutateTensor)
{
	vext::Tensor<float> sigmoid_tensor({ 0.0f, 2.0f });
	sigmoid_tensor.sigmoid();
	expect_tensor_near(sigmoid_tensor, { 0.5f, 1.0f / (1.0f + std::exp(-2.0f)) });

	vext::Tensor<float> soft_relu_tensor({ 0.0f, 2.0f });
	soft_relu_tensor.soft_relu();
	expect_tensor_near(soft_relu_tensor, { std::log(2.0f), std::log(1.0f + std::exp(2.0f)) });

	vext::Tensor<float> relu_tensor({ -2.0f, 0.0f, 3.0f });
	relu_tensor.relu();
	expect_tensor_near(relu_tensor, { 0.0f, 0.0f, 3.0f });

	vext::Tensor<float> leaky_relu_tensor({ -2.0f, 3.0f });
	leaky_relu_tensor.leaky_relu(0.25f);
	expect_tensor_near(leaky_relu_tensor, { -0.5f, 3.0f });

	vext::Tensor<float> elu_tensor({ -1.0f, 2.0f });
	elu_tensor.elu(2.0f);
	expect_tensor_near(elu_tensor, { 2.0f * (std::exp(-1.0f) - 1.0f), 2.0f });

	vext::Tensor<float> swish_tensor({ -1.0f, 2.0f });
	swish_tensor.swish(1.0f);
	expect_tensor_near(swish_tensor, { -1.0f / (1.0f + std::exp(1.0f)), 2.0f / (1.0f + std::exp(-2.0f)) });
}

TEST(TensorCpu, NormalizationUnaryOperationsMutateTensor)
{
	vext::Tensor<float> softmax_tensor({ 1.0f, 2.0f, 3.0f });
	softmax_tensor.softmax();
	const float softmax_sum = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);
	expect_tensor_near(softmax_tensor, { std::exp(1.0f) / softmax_sum, std::exp(2.0f) / softmax_sum, std::exp(3.0f) / softmax_sum });

	vext::Tensor<float> softmin_tensor({ 1.0f, 2.0f, 3.0f });
	softmin_tensor.softmin();
	const float softmin_sum = std::exp(-1.0f) + std::exp(-2.0f) + std::exp(-3.0f);
	expect_tensor_near(softmin_tensor, { std::exp(-1.0f) / softmin_sum, std::exp(-2.0f) / softmin_sum, std::exp(-3.0f) / softmin_sum });

	vext::Tensor<float> log_softmax_tensor({ 1.0f, 2.0f, 3.0f });
	log_softmax_tensor.log_softmax();
	expect_tensor_near(log_softmax_tensor, { std::log(std::exp(1.0f) / softmax_sum), std::log(std::exp(2.0f) / softmax_sum), std::log(std::exp(3.0f) / softmax_sum) });
}

TEST(TensorCpu, ParameterizedUnaryOperationsMutateTensor)
{
	vext::Tensor<float> linear_tensor({ -1.0f, 2.0f });
	linear_tensor.linear(2.0f, 3.0f);
	expect_tensor_near(linear_tensor, { 1.0f, 7.0f });

	vext::Tensor<float> clip_tensor({ -2.0f, 0.5f, 3.0f });
	clip_tensor.clip(-1.0f, 1.0f);
	expect_tensor_near(clip_tensor, { -1.0f, 0.5f, 1.0f });

	vext::Tensor<float> pow_tensor({ 2.0f, 3.0f });
	pow_tensor.pow(2.0f, 3.0f);
	expect_tensor_near(pow_tensor, { 16.0f, 54.0f });
}

TEST(TensorCpu, UnaryMinusNegatesInPlace)
{
	vext::Tensor<std::int32_t> tensor({ 1, -2, 3 });

	-tensor;

	expect_tensor_values(tensor, { -1, 2, -3 });
}

TEST(TensorCpu, ReductionsCollapseWholeTensorWhenNoAxisIsProvided)
{
	const vext::Tensor<float> tensor({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

	const vext::Tensor<float> sum      = tensor.sum();
	const vext::Tensor<float> product  = tensor.prod();
	const vext::Tensor<float> minimum  = tensor.min();
	const vext::Tensor<float> maximum  = tensor.max();
	const vext::Tensor<float> mean     = tensor.mean();
	const vext::Tensor<float> variance = tensor.var();
	const vext::Tensor<float> stddev   = tensor.std();

	expect_scalar_near(sum, 21.0f);
	expect_scalar_near(product, 720.0f);
	expect_scalar_near(minimum, 1.0f);
	expect_scalar_near(maximum, 6.0f);
	expect_scalar_near(mean, 3.5f);
	expect_scalar_near(variance, 17.5f / 6.0f);
	expect_scalar_near(stddev, std::sqrt(17.5f / 6.0f));
}

TEST(TensorCpu, ReductionsSupportSingleAxis)
{
	const vext::Tensor<float> tensor({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

	const vext::Tensor<float> column_sum  = tensor.sum(0);
	const vext::Tensor<float> row_sum     = tensor.sum(1);
	const vext::Tensor<float> column_mean = tensor.mean(0);
	const vext::Tensor<float> row_mean    = tensor.mean(1);
	const vext::Tensor<float> column_min  = tensor.min(0);
	const vext::Tensor<float> row_max     = tensor.max(1);

	expect_shape_eq(column_sum.shape(), { 3 });
	expect_tensor_near(column_sum, { 5.0f, 7.0f, 9.0f });

	expect_shape_eq(row_sum.shape(), { 2 });
	expect_tensor_near(row_sum, { 6.0f, 15.0f });

	expect_tensor_near(column_mean, { 2.5f, 3.5f, 4.5f });
	expect_tensor_near(row_mean, { 2.0f, 5.0f });
	expect_tensor_near(column_min, { 1.0f, 2.0f, 3.0f });
	expect_tensor_near(row_max, { 3.0f, 6.0f });
}

TEST(TensorCpu, ReductionsSupportMultipleAxes)
{
	const vext::Tensor<float> tensor({ { { 1.0f, 2.0f }, { 3.0f, 4.0f } }, { { 5.0f, 6.0f }, { 7.0f, 8.0f } } });

	const vext::Tensor<float> result = tensor.sum(1, 2);

	expect_shape_eq(result.shape(), { 2 });
	expect_tensor_near(result, { 10.0f, 26.0f });
}

TEST(TensorCpu, ReductionsRejectInvalidAxes)
{
	const vext::Tensor<float> tensor({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });

	EXPECT_THROW((void)tensor.sum(2), std::runtime_error);
	EXPECT_THROW((void)tensor.sum(0, 0), std::runtime_error);
	EXPECT_THROW((void)tensor.sum(0, 1, 2), std::runtime_error);
	EXPECT_THROW((void)tensor.sum(-1), std::runtime_error);
}

TEST(TensorCpu, MatmulComputesMatrixProduct)
{
	const vext::Tensor<float> lhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float> rhs({ { 7.0f, 8.0f }, { 9.0f, 10.0f }, { 11.0f, 12.0f } });

	const vext::Tensor<float> result = lhs.matmul(rhs);

	expect_shape_eq(result.shape(), { 2, 2 });
	expect_tensor_near(result, { 58.0f, 64.0f, 139.0f, 154.0f });
}

TEST(TensorCpu, MatmulSupportsBatchedLeftHandTensor)
{
	const vext::Tensor<float> lhs({ { { 1.0f, 2.0f }, { 3.0f, 4.0f } }, { { 5.0f, 6.0f }, { 7.0f, 8.0f } } });
	const vext::Tensor<float> rhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

	const vext::Tensor<float> result = lhs.matmul(rhs);

	expect_shape_eq(result.shape(), { 2, 2, 3 });
	expect_tensor_near(result, { 9.0f, 12.0f, 15.0f, 19.0f, 26.0f, 33.0f, 29.0f, 40.0f, 51.0f, 39.0f, 54.0f, 69.0f });
}

TEST(TensorCpu, MatmulRejectsIncompatibleShapes)
{
	const vext::Tensor<float> lhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float> rhs({ { 1.0f, 2.0f }, { 3.0f, 4.0f }, { 5.0f, 6.0f }, { 7.0f, 8.0f } });

	EXPECT_THROW((void)lhs.matmul(rhs), std::runtime_error);
}
