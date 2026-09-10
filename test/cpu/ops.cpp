#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <numbers>
#include <type_traits>
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

template <typename Tp>
void
expect_scalar_eq(
	const vext::Tensor<Tp, vext::Backend::CPU>& tensor,
	const Tp                                    expected)
{
	expect_shape_eq(tensor.dims(), { 1 });
	EXPECT_EQ(tensor.item(0), expected);
}

template <typename Tp>
void
expect_scalar_near(
	const vext::Tensor<Tp, vext::Backend::CPU>& tensor,
	const Tp                                    expected,
	const Tp                                    tolerance = static_cast<Tp>(1e-5))
{
	expect_shape_eq(tensor.dims(), { 1 });
	EXPECT_NEAR(tensor.item(0), expected, tolerance);
}

}

TEST(TensorCpu, LogicalComparisonsProduceMaskTensors)
{
	const vext::Tensor<std::int32_t> lhs({ 1, 2, 3, 4 });
	const vext::Tensor<std::int32_t> rhs({ 1, 0, 3, 5 });

	expect_tensor_values(vext::ops::logical<vext::LogicOp::EQUAL>(lhs, rhs), { 1, 0, 1, 0 });
	expect_tensor_values(vext::ops::logical<vext::LogicOp::NOT_EQUAL>(lhs, rhs), { 0, 1, 0, 1 });
	expect_tensor_values(vext::ops::logical<vext::LogicOp::LESS>(lhs, rhs), { 0, 0, 0, 1 });
	expect_tensor_values(vext::ops::logical<vext::LogicOp::LESS_EQUAL>(lhs, rhs), { 1, 0, 1, 1 });
	expect_tensor_values(vext::ops::logical<vext::LogicOp::GREATER>(lhs, rhs), { 0, 1, 0, 0 });
	expect_tensor_values(vext::ops::logical<vext::LogicOp::GREATER_EQUAL>(lhs, rhs), { 1, 1, 1, 0 });
}

TEST(TensorCpu, LogicalComparisonsRejectIncompatibleShapes)
{
	const vext::Tensor<float> lhs({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });
	const vext::Tensor<float> rhs({ 1.0f, 2.0f });

	EXPECT_THROW((void)(vext::ops::logical<vext::LogicOp::EQUAL>(lhs, rhs)), std::runtime_error);
}

TEST(TensorCpu, ElementwiseArithmeticSupportsSameShape)
{
	const vext::Tensor<std::int32_t> lhs({ 8, 12, 20 });
	const vext::Tensor<std::int32_t> rhs({ 2, 3, 4 });

	const vext::Tensor<std::int32_t> sum        = vext::ops::binary<vext::BinaryOp::ADD>(lhs, rhs);
	const vext::Tensor<std::int32_t> difference = vext::ops::binary<vext::BinaryOp::SUB>(lhs, rhs);
	const vext::Tensor<std::int32_t> product    = vext::ops::binary<vext::BinaryOp::MUL>(lhs, rhs);
	const vext::Tensor<std::int32_t> quotient   = vext::ops::binary<vext::BinaryOp::DIV>(lhs, rhs);
	const vext::Tensor<std::int32_t> power      = vext::ops::binary<vext::BinaryOp::POW>(rhs, rhs);

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

	const vext::Tensor<float> result = vext::ops::binary<vext::BinaryOp::ADD>(lhs, rhs);

	static_assert(std::is_same_v<decltype(result), const vext::Tensor<float>>);
	expect_tensor_near(result, { 1.5f, 3.25f });
}

TEST(TensorCpu, ElementwiseArithmeticBroadcastsRightHandTensor)
{
	const vext::Tensor<float> matrix({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float> bias({ 10.0f, 20.0f, 30.0f });

	const vext::Tensor<float> result = vext::ops::binary<vext::BinaryOp::ADD>(matrix, bias);

	expect_shape_eq(result.dims(), { 2, 3 });
	expect_tensor_near(result, { 11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f });
}

TEST(TensorCpu, ElementwiseArithmeticRejectsIncompatibleShapes)
{
	const vext::Tensor<float> lhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float> rhs({ 1.0f, 2.0f, 3.0f, 4.0f });

	EXPECT_THROW((void)(vext::ops::binary<vext::BinaryOp::ADD>(lhs, rhs)), std::runtime_error);
}

TEST(TensorCpu, PreluMutatesTensorWithElementwiseSlope)
{
	vext::Tensor<float>       values({ -2.0f, -1.0f, 0.0f, 3.0f });
	const vext::Tensor<float> slopes({ 0.25f, 0.5f, 0.75f, 1.0f });

	vext::ops::binary<vext::BinaryOp::PRELU>(values, slopes);

	expect_tensor_near(values, { -0.5f, -0.5f, 0.0f, 3.0f });
}

TEST(TensorCpu, ParameterlessUnaryOpsMutateTensor)
{
	vext::Tensor<float> abs_tensor({ -1.0f, 0.0f, 4.0f });
	vext::ops::unary<vext::UnaryOp::ABS>(abs_tensor);
	expect_tensor_near(abs_tensor, { 1.0f, 0.0f, 4.0f });

	vext::Tensor<float> sin_tensor({ 0.0f, static_cast<float>(std::numbers::pi / 2.0) });
	vext::ops::unary<vext::UnaryOp::SIN>(sin_tensor);
	expect_tensor_near(sin_tensor, { 0.0f, 1.0f });

	vext::Tensor<float> cos_tensor({ 0.0f, static_cast<float>(std::numbers::pi) });
	vext::ops::unary<vext::UnaryOp::COS>(cos_tensor);
	expect_tensor_near(cos_tensor, { 1.0f, -1.0f });

	vext::Tensor<float> exp_tensor({ 0.0f, 1.0f });
	vext::ops::unary<vext::UnaryOp::EXP>(exp_tensor);
	expect_tensor_near(exp_tensor, { 1.0f, std::exp(1.0f) });

	vext::Tensor<float> log_tensor({ 1.0f, std::exp(2.0f) });
	vext::ops::unary<vext::UnaryOp::LOG>(log_tensor);
	expect_tensor_near(log_tensor, { 0.0f, 2.0f });

	vext::Tensor<float> sqrt_tensor({ 1.0f, 4.0f, 9.0f });
	vext::ops::unary<vext::UnaryOp::SQRT>(sqrt_tensor);
	expect_tensor_near(sqrt_tensor, { 1.0f, 2.0f, 3.0f });

	vext::Tensor<float> square_tensor({ -2.0f, 3.0f });
	vext::ops::unary<vext::UnaryOp::SQUARE>(square_tensor);
	expect_tensor_near(square_tensor, { 4.0f, 9.0f });

	vext::Tensor<float> round_tensor({ 1.2f, 1.5f, -1.6f });
	vext::ops::unary<vext::UnaryOp::ROUND>(round_tensor);
	expect_tensor_near(round_tensor, { 1.0f, 2.0f, -2.0f });
}

TEST(TensorCpu, ActivationUnaryOpsMutateTensor)
{
	vext::Tensor<float> sigmoid_tensor({ 0.0f, 2.0f });
	vext::ops::unary<vext::UnaryOp::SIGMOID>(sigmoid_tensor);
	expect_tensor_near(sigmoid_tensor, { 0.5f, 1.0f / (1.0f + std::exp(-2.0f)) });

	vext::Tensor<float> soft_relu_tensor({ 0.0f, 2.0f });
	vext::ops::unary<vext::UnaryOp::SOFT_RELU>(soft_relu_tensor);
	expect_tensor_near(soft_relu_tensor, { std::log(2.0f), std::log(1.0f + std::exp(2.0f)) });

	vext::Tensor<float> relu_tensor({ -2.0f, 0.0f, 3.0f });
	vext::ops::unary<vext::UnaryOp::RELU>(relu_tensor);
	expect_tensor_near(relu_tensor, { 0.0f, 0.0f, 3.0f });

	vext::Tensor<float> leaky_relu_tensor({ -2.0f, 3.0f });
	vext::ops::unary<vext::UnaryOp::LEAKY_RELU>(leaky_relu_tensor, 0.25f);
	expect_tensor_near(leaky_relu_tensor, { -0.5f, 3.0f });

	vext::Tensor<float> elu_tensor({ -1.0f, 2.0f });
	vext::ops::unary<vext::UnaryOp::ELU>(elu_tensor, 2.0f);
	expect_tensor_near(elu_tensor, { 2.0f * (std::exp(-1.0f) - 1.0f), 2.0f });

	vext::Tensor<float> swish_tensor({ -1.0f, 2.0f });
	vext::ops::unary<vext::UnaryOp::SWISH>(swish_tensor, 1.0f);
	expect_tensor_near(swish_tensor, { -1.0f / (1.0f + std::exp(1.0f)), 2.0f / (1.0f + std::exp(-2.0f)) });
}

TEST(TensorCpu, NormalizationUnaryOpsMutateTensor)
{
	vext::Tensor<float> softmax_tensor({ 1.0f, 2.0f, 3.0f });
	vext::ops::unary<vext::UnaryOp::SOFTMAX>(softmax_tensor);
	const float softmax_sum = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);
	expect_tensor_near(softmax_tensor, { std::exp(1.0f) / softmax_sum, std::exp(2.0f) / softmax_sum, std::exp(3.0f) / softmax_sum });

	vext::Tensor<float> softmin_tensor({ 1.0f, 2.0f, 3.0f });
	vext::ops::unary<vext::UnaryOp::SOFTMIN>(softmin_tensor);
	const float softmin_sum = std::exp(-1.0f) + std::exp(-2.0f) + std::exp(-3.0f);
	expect_tensor_near(softmin_tensor, { std::exp(-1.0f) / softmin_sum, std::exp(-2.0f) / softmin_sum, std::exp(-3.0f) / softmin_sum });

	vext::Tensor<float> log_softmax_tensor({ 1.0f, 2.0f, 3.0f });
	vext::ops::unary<vext::UnaryOp::LOGSOFTMAX>(log_softmax_tensor);
	expect_tensor_near(log_softmax_tensor, { std::log(std::exp(1.0f) / softmax_sum), std::log(std::exp(2.0f) / softmax_sum), std::log(std::exp(3.0f) / softmax_sum) });
}

TEST(TensorCpu, ParameterizedUnaryOpsMutateTensor)
{
	vext::Tensor<float> linear_tensor({ -1.0f, 2.0f });
	vext::ops::unary<vext::UnaryOp::LINEAR>(linear_tensor, 2.0f, 3.0f);
	expect_tensor_near(linear_tensor, { 1.0f, 7.0f });

	vext::Tensor<float> clip_tensor({ -2.0f, 0.5f, 3.0f });
	vext::ops::unary<vext::UnaryOp::CLIP>(clip_tensor, -1.0f, 1.0f);
	expect_tensor_near(clip_tensor, { -1.0f, 0.5f, 1.0f });

	vext::Tensor<float> pow_tensor({ 2.0f, 3.0f });
	vext::ops::unary<vext::UnaryOp::CLIP>(pow_tensor, 2.0f, 3.0f);
	expect_tensor_near(pow_tensor, { 16.0f, 54.0f });
}

TEST(TensorCpu, UnaryMinusNegatesInPlace)
{
	vext::Tensor<std::int32_t> tensor({ 1, -2, 3 });
	vext::ops::unary<vext::UnaryOp::NEG>(tensor);

	expect_tensor_values(tensor, { -1, 2, -3 });
}

TEST(TensorCpu, SupportsWholeTensorReductions)
{
	const vext::Tensor<float> tensor({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

	const vext::Tensor<float> sum      = vext::ops::reduction<vext::ReductionOp::SUM>(tensor);
	const vext::Tensor<float> product  = vext::ops::reduction<vext::ReductionOp::PROD>(tensor);
	const vext::Tensor<float> minimum  = vext::ops::reduction<vext::ReductionOp::MIN>(tensor);
	const vext::Tensor<float> maximum  = vext::ops::reduction<vext::ReductionOp::MAX>(tensor);
	const vext::Tensor<float> mean     = vext::ops::reduction<vext::ReductionOp::MEAN>(tensor);
	const vext::Tensor<float> variance = vext::ops::reduction<vext::ReductionOp::VAR>(tensor);
	const vext::Tensor<float> stddev   = vext::ops::reduction<vext::ReductionOp::STD>(tensor);

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

	const vext::Tensor<float> column_sum  = vext::ops::reduction<vext::ReductionOp::SUM>(tensor, 0);
	const vext::Tensor<float> row_sum     = vext::ops::reduction<vext::ReductionOp::SUM>(tensor, 1);
	const vext::Tensor<float> column_mean = vext::ops::reduction<vext::ReductionOp::MEAN>(tensor, 0);
	const vext::Tensor<float> row_mean    = vext::ops::reduction<vext::ReductionOp::MEAN>(tensor, 1);
	const vext::Tensor<float> column_min  = vext::ops::reduction<vext::ReductionOp::MIN>(tensor, 0);
	const vext::Tensor<float> row_max     = vext::ops::reduction<vext::ReductionOp::MAX>(tensor, 1);

	expect_shape_eq(column_sum.dims(), { 3 });
	expect_tensor_near(column_sum, { 5.0f, 7.0f, 9.0f });

	expect_shape_eq(row_sum.dims(), { 2 });
	expect_tensor_near(row_sum, { 6.0f, 15.0f });

	expect_tensor_near(column_mean, { 2.5f, 3.5f, 4.5f });
	expect_tensor_near(row_mean, { 2.0f, 5.0f });
	expect_tensor_near(column_min, { 1.0f, 2.0f, 3.0f });
	expect_tensor_near(row_max, { 3.0f, 6.0f });
}

TEST(TensorCpu, ReductionsSupportMultipleAxes)
{
	const vext::Tensor<float> tensor({ { { 1.0f, 2.0f }, { 3.0f, 4.0f } }, { { 5.0f, 6.0f }, { 7.0f, 8.0f } } });

	const vext::Tensor<float> result = vext::ops::reduction<vext::ReductionOp::SUM>(tensor, 1, 2);

	expect_shape_eq(result.dims(), { 2 });
	expect_tensor_near(result, { 10.0f, 26.0f });
}

TEST(TensorCpu, ReductionsRejectInvalidAxes)
{
	const vext::Tensor<float> tensor({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });

	EXPECT_THROW((void)vext::ops::reduction<vext::ReductionOp::SUM>(tensor, 2), std::runtime_error);
	EXPECT_THROW((void)vext::ops::reduction<vext::ReductionOp::SUM>(tensor, 0, 0), std::runtime_error);
	EXPECT_THROW((void)vext::ops::reduction<vext::ReductionOp::SUM>(tensor, 0, 1, 2), std::runtime_error);
	EXPECT_THROW((void)vext::ops::reduction<vext::ReductionOp::SUM>(tensor, -1), std::runtime_error);
}

TEST(TensorCpu, CsrScatterAggregatesNeighborRows)
{
	const vext::Tensor<float>         src({ { 1.0f, 2.0f }, { 3.0f, 4.0f }, { -2.0f, 5.0f }, { 4.0f, -1.0f } });
	const vext::Tensor<std::uint32_t> head({ 0U, 2U, 4U, 4U, 4U });
	const vext::Tensor<std::uint32_t> tail({ 0U, 2U, 1U, 3U });

	const vext::Tensor<float> sum      = vext::ops::csr_scatter<vext::CSRScatterOp::SUM>(src, head, tail);
	const vext::Tensor<float> mean     = vext::ops::csr_scatter<vext::CSRScatterOp::MEAN>(src, head, tail);
	const vext::Tensor<float> maximum  = vext::ops::csr_scatter<vext::CSRScatterOp::MAX>(src, head, tail);
	const vext::Tensor<float> variance = vext::ops::csr_scatter<vext::CSRScatterOp::VAR>(src, head, tail);
	const vext::Tensor<float> stddev   = vext::ops::csr_scatter<vext::CSRScatterOp::STD>(src, head, tail);

	expect_shape_eq(sum.dims(), { 4, 2 });
	expect_shape_eq(mean.dims(), { 4, 2 });
	expect_shape_eq(maximum.dims(), { 4, 2 });
	expect_shape_eq(variance.dims(), { 4, 2 });
	expect_shape_eq(stddev.dims(), { 4, 2 });

	expect_tensor_near(sum, { -1.0f, 7.0f, 7.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f });
	expect_tensor_near(mean, { -0.5f, 3.5f, 3.5f, 1.5f, 0.0f, 0.0f, 0.0f, 0.0f });
	expect_tensor_near(maximum, { 1.0f, 5.0f, 4.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f });
	expect_tensor_near(variance, { 2.25f, 2.25f, 0.25f, 6.25f, 0.0f, 0.0f, 0.0f, 0.0f });
	expect_tensor_near(stddev, { 1.5f, 1.5f, 0.5f, 2.5f, 0.0f, 0.0f, 0.0f, 0.0f });
}

TEST(TensorCpu, CsrScatterMinAndProdUseOpIdentity)
{
	const vext::Tensor<float>         src({ { 1.0f, 2.0f }, { 3.0f, 4.0f }, { -2.0f, 5.0f }, { 4.0f, -1.0f } });
	const vext::Tensor<std::uint32_t> head({ 0U, 2U, 4U, 4U, 4U });
	const vext::Tensor<std::uint32_t> tail({ 0U, 2U, 1U, 3U });

	const vext::Tensor<float> minimum = vext::ops::csr_scatter<vext::CSRScatterOp::MIN>(src, head, tail);
	const vext::Tensor<float> product = vext::ops::csr_scatter<vext::CSRScatterOp::PROD>(src, head, tail);

	expect_shape_eq(minimum.dims(), { 4, 2 });
	expect_shape_eq(product.dims(), { 4, 2 });

	expect_tensor_near(minimum, { -2.0f, 2.0f, 3.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f });
	expect_tensor_near(product, { -2.0f, 10.0f, 12.0f, -4.0f, 0.0f, 0.0f, 0.0f, 0.0f });
}

TEST(TensorCpu, CsrSpmvAggregatesSparseMatrixVectorProducts)
{
	const vext::Tensor<float>         values({ 1.0f, 2.0f, 3.0f, 4.0f, -2.0f, 5.0f });
	const vext::Tensor<std::uint32_t> head({ 0U, 2U, 4U, 6U });
	const vext::Tensor<std::uint32_t> tail({ 0U, 2U, 1U, 3U, 0U, 1U });
	const vext::Tensor<float>         x({ 2.0f, -1.0f, 3.0f, 4.0f });

	const vext::Tensor<float> sum      = vext::ops::csr_spmv<vext::CSRSpMVOp::SUM>(values, head, tail, x);
	const vext::Tensor<float> mean     = vext::ops::csr_spmv<vext::CSRSpMVOp::SUM>(values, head, tail, x);
	const vext::Tensor<float> minimum  = vext::ops::csr_spmv<vext::CSRSpMVOp::SUM>(values, head, tail, x);
	const vext::Tensor<float> maximum  = vext::ops::csr_spmv<vext::CSRSpMVOp::SUM>(values, head, tail, x);
	const vext::Tensor<float> product  = vext::ops::csr_spmv<vext::CSRSpMVOp::SUM>(values, head, tail, x);
	const vext::Tensor<float> variance = vext::ops::csr_spmv<vext::CSRSpMVOp::SUM>(values, head, tail, x);
	const vext::Tensor<float> stddev   = vext::ops::csr_spmv<vext::CSRSpMVOp::SUM>(values, head, tail, x);

	expect_shape_eq(sum.dims(), { 3 });
	expect_shape_eq(mean.dims(), { 3 });
	expect_shape_eq(minimum.dims(), { 3 });
	expect_shape_eq(maximum.dims(), { 3 });
	expect_shape_eq(product.dims(), { 3 });
	expect_shape_eq(variance.dims(), { 3 });
	expect_shape_eq(stddev.dims(), { 3 });

	expect_tensor_near(sum, { 8.0f, 13.0f, -9.0f });
	expect_tensor_near(mean, { 4.0f, 6.5f, -4.5f });
	expect_tensor_near(minimum, { 2.0f, -3.0f, -5.0f });
	expect_tensor_near(maximum, { 6.0f, 16.0f, -4.0f });
	expect_tensor_near(product, { 12.0f, -48.0f, 20.0f });
	expect_tensor_near(variance, { 4.0f, 90.25f, 0.25f });
	expect_tensor_near(stddev, { 2.0f, 9.5f, 0.5f });
}

TEST(TensorCpu, MatmulComputesMatrixProduct)
{
	const vext::Tensor<float> lhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float> rhs({ { 7.0f, 8.0f }, { 9.0f, 10.0f }, { 11.0f, 12.0f } });

	const vext::Tensor<float> result = vext::ops::matmul(lhs, rhs);

	expect_shape_eq(result.dims(), { 2, 2 });
	expect_tensor_near(result, { 58.0f, 64.0f, 139.0f, 154.0f });
}

TEST(TensorCpu, MatmulSupportsBatchedLeftHandTensor)
{
	const vext::Tensor<float> lhs({ { { 1.0f, 2.0f }, { 3.0f, 4.0f } }, { { 5.0f, 6.0f }, { 7.0f, 8.0f } } });
	const vext::Tensor<float> rhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

	const vext::Tensor<float> result = vext::ops::matmul(lhs, rhs);

	expect_shape_eq(result.dims(), { 2, 2, 3 });
	expect_tensor_near(result, { 9.0f, 12.0f, 15.0f, 19.0f, 26.0f, 33.0f, 29.0f, 40.0f, 51.0f, 39.0f, 54.0f, 69.0f });
}

TEST(TensorCpu, MatmulRejectsIncompatibleShapes)
{
	const vext::Tensor<float> lhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float> rhs({ { 1.0f, 2.0f }, { 3.0f, 4.0f }, { 5.0f, 6.0f }, { 7.0f, 8.0f } });

	EXPECT_THROW((void)vext::ops::matmul(lhs, rhs), std::runtime_error);
}
