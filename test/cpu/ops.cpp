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

float
noise_value(
	const std::uint32_t seed,
	const std::uint32_t counter,
	const std::uint32_t index,
	const std::int8_t   direction = 1)
{
	std::uint32_t hash = seed ^ (counter * 0x85ebca6bu) ^ (index * 0x9e3779b9u);
	hash ^= hash >> 16;
	hash *= 0x7feb352du;
	hash ^= hash >> 15;
	hash *= 0x846ca68bu;
	hash ^= hash >> 16;

	return static_cast<float>(hash >> 8) * (1.0f / 16777216.0f) * direction;
}

void
set_noise_descriptor(
	const std::uint32_t seed,
	const std::uint32_t counter,
	const std::int8_t   direction = 1)
{
	vext::core::cpu::NoiseDescriptor& descriptor = vext::core::cpu::sequentional_noise_descriptor();
	descriptor.seed                              = seed;
	descriptor.counter                           = counter;
	descriptor.direction                         = direction;
}

template <vext::Op Kp>
concept UnaryCallable = requires(vext::Tensor<float>& tensor) {
	vext::unary<Kp>(tensor, vext::values({}), tensor);
};

template <vext::Op Kp>
concept BinaryCallable = requires(const vext::Tensor<float>& lhs, const vext::Tensor<float>& rhs) {
	vext::binary<Kp>(lhs, rhs);
};

template <vext::Op Kp>
concept LogicalCallable = requires(const vext::Tensor<float>& lhs, const vext::Tensor<float>& rhs) {
	vext::logical<Kp>(lhs, rhs);
};

template <vext::Op Kp>
concept ReductionCallable = requires(const vext::Tensor<float>& tensor) {
	vext::reduction<Kp>(tensor);
};

static_assert(UnaryCallable<vext::Op::ABS>);
static_assert(!UnaryCallable<vext::Op::SUM>);
static_assert(BinaryCallable<vext::Op::ADD>);
static_assert(!BinaryCallable<vext::Op::ABS>);
static_assert(LogicalCallable<vext::Op::EQUAL>);
static_assert(!LogicalCallable<vext::Op::ADD>);
static_assert(ReductionCallable<vext::Op::SUM>);
static_assert(!ReductionCallable<vext::Op::EQUAL>);

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

	expect_tensor_values(vext::logical<vext::Op::EQUAL>(lhs, rhs), { 1, 0, 1, 0 });
	expect_tensor_values(vext::logical<vext::Op::NOT_EQUAL>(lhs, rhs), { 0, 1, 0, 1 });
	expect_tensor_values(vext::logical<vext::Op::LESS>(lhs, rhs), { 0, 0, 0, 1 });
	expect_tensor_values(vext::logical<vext::Op::LESS_EQUAL>(lhs, rhs), { 1, 0, 1, 1 });
	expect_tensor_values(vext::logical<vext::Op::GREATER>(lhs, rhs), { 0, 1, 0, 0 });
	expect_tensor_values(vext::logical<vext::Op::GREATER_EQUAL>(lhs, rhs), { 1, 1, 1, 0 });
}

TEST(TensorCpu, LogicalComparisonsRejectIncompatibleShapes)
{
	const vext::Tensor<float> lhs({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });
	const vext::Tensor<float> rhs({ 1.0f, 2.0f });

	EXPECT_THROW((void)(vext::logical<vext::Op::EQUAL>(lhs, rhs)), std::runtime_error);
}

TEST(TensorCpu, OperationsRejectEmptyInputTensors)
{
	vext::Tensor<float>         empty;
	vext::Tensor<std::uint32_t> empty_indices;

	const vext::Tensor<float>         values({ 1.0f, 2.0f });
	const vext::Tensor<float>         matrix({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });
	const vext::Tensor<std::uint32_t> head({ 0U, 1U, 2U });
	const vext::Tensor<std::uint32_t> tail({ 0U, 1U });

	EXPECT_THROW(vext::unary<vext::Op::RELU>(empty, vext::values({}), empty), std::runtime_error);

	EXPECT_THROW((void)vext::binary<vext::Op::ADD>(empty, values), std::runtime_error);
	EXPECT_THROW((void)vext::binary<vext::Op::ADD>(values, empty), std::runtime_error);

	EXPECT_THROW((void)vext::logical<vext::Op::EQUAL>(empty, values), std::runtime_error);
	EXPECT_THROW((void)vext::logical<vext::Op::EQUAL>(values, empty), std::runtime_error);

	EXPECT_THROW((void)vext::reduction<vext::Op::SUM>(empty), std::runtime_error);

	EXPECT_THROW((void)vext::csr_scatter<vext::Op::SUM>(empty, head, tail), std::runtime_error);
	EXPECT_THROW((void)vext::csr_scatter<vext::Op::SUM>(matrix, empty_indices, tail), std::runtime_error);
	EXPECT_THROW((void)vext::csr_scatter<vext::Op::SUM>(matrix, head, empty_indices), std::runtime_error);

	EXPECT_THROW((void)vext::csr_spmv<vext::Op::SUM>(empty, head, tail, values), std::runtime_error);
	EXPECT_THROW((void)vext::csr_spmv<vext::Op::SUM>(values, empty_indices, tail, values), std::runtime_error);
	EXPECT_THROW((void)vext::csr_spmv<vext::Op::SUM>(values, head, empty_indices, values), std::runtime_error);
	EXPECT_THROW((void)vext::csr_spmv<vext::Op::SUM>(values, head, tail, empty), std::runtime_error);

	EXPECT_THROW((void)vext::matmul(empty, matrix), std::runtime_error);
	EXPECT_THROW((void)vext::matmul(matrix, empty), std::runtime_error);
}

TEST(TensorCpu, ElementwiseArithmeticSupportsSameShape)
{
	const vext::Tensor<std::int32_t> lhs({ 8, 12, 20 });
	const vext::Tensor<std::int32_t> rhs({ 2, 3, 4 });

	const vext::Tensor<std::int32_t> sum        = vext::binary<vext::Op::ADD>(lhs, rhs);
	const vext::Tensor<std::int32_t> difference = vext::binary<vext::Op::SUB>(lhs, rhs);
	const vext::Tensor<std::int32_t> product    = vext::binary<vext::Op::MUL>(lhs, rhs);
	const vext::Tensor<std::int32_t> quotient   = vext::binary<vext::Op::DIV>(lhs, rhs);
	const vext::Tensor<std::int32_t> power      = vext::binary<vext::Op::POW>(rhs, rhs);

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

	const vext::Tensor<float> result = vext::binary<vext::Op::ADD>(lhs, rhs);

	static_assert(std::is_same_v<decltype(result), const vext::Tensor<float>>);
	expect_tensor_near(result, { 1.5f, 3.25f });
}

TEST(TensorCpu, ElementwiseArithmeticBroadcastsRightHandTensor)
{
	const vext::Tensor<float> matrix({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float> bias({ 10.0f, 20.0f, 30.0f });

	const vext::Tensor<float> result = vext::binary<vext::Op::ADD>(matrix, bias);

	expect_shape_eq(result.dims(), { 2, 3 });
	expect_tensor_near(result, { 11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f });
}

TEST(TensorCpu, ElementwiseArithmeticRejectsIncompatibleShapes)
{
	const vext::Tensor<float> lhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float> rhs({ 1.0f, 2.0f, 3.0f, 4.0f });

	EXPECT_THROW((void)(vext::binary<vext::Op::ADD>(lhs, rhs)), std::runtime_error);
}

TEST(TensorCpu, PreluMutatesTensorWithElementwiseSlope)
{
	vext::Tensor<float>       values({ -2.0f, -1.0f, 0.0f, 3.0f });
	const vext::Tensor<float> slopes({ 0.25f, 0.5f, 0.75f, 1.0f });

	vext::binary<vext::Op::PRELU>(values, slopes, values);

	expect_tensor_near(values, { -0.5f, -0.5f, 0.0f, 3.0f });
}

TEST(TensorCpu, PreluReturnedOutputLeavesInputsUnchanged)
{
	const vext::Tensor<float> values({ -2.0f, -1.0f, 0.0f, 3.0f });
	const vext::Tensor<float> slopes({ 0.25f, 0.5f, 0.75f, 1.0f });

	const vext::Tensor<float> result = vext::binary<vext::Op::PRELU>(values, slopes);

	expect_tensor_near(result, { -0.5f, -0.5f, 0.0f, 3.0f });
	expect_tensor_near(values, { -2.0f, -1.0f, 0.0f, 3.0f });
}

TEST(TensorCpu, ParameterlessUnaryOpsMutateTensor)
{
	vext::Tensor<float> abs_tensor({ -1.0f, 0.0f, 4.0f });
	vext::unary<vext::Op::ABS>(abs_tensor, vext::values({}), abs_tensor);
	expect_tensor_near(abs_tensor, { 1.0f, 0.0f, 4.0f });

	vext::Tensor<float> sin_tensor({ 0.0f, static_cast<float>(std::numbers::pi / 2.0) });
	vext::unary<vext::Op::SIN>(sin_tensor, vext::values({}), sin_tensor);
	expect_tensor_near(sin_tensor, { 0.0f, 1.0f });

	vext::Tensor<float> cos_tensor({ 0.0f, static_cast<float>(std::numbers::pi) });
	vext::unary<vext::Op::COS>(cos_tensor, vext::values({}), cos_tensor);
	expect_tensor_near(cos_tensor, { 1.0f, -1.0f });

	vext::Tensor<float> exp_tensor({ 0.0f, 1.0f });
	vext::unary<vext::Op::EXP>(exp_tensor, vext::values({}), exp_tensor);
	expect_tensor_near(exp_tensor, { 1.0f, std::exp(1.0f) });

	vext::Tensor<float> log_tensor({ 1.0f, std::exp(2.0f) });
	vext::unary<vext::Op::LOG>(log_tensor, vext::values({}), log_tensor);
	expect_tensor_near(log_tensor, { 0.0f, 2.0f });

	vext::Tensor<float> sqrt_tensor({ 1.0f, 4.0f, 9.0f });
	vext::unary<vext::Op::SQRT>(sqrt_tensor, vext::values({}), sqrt_tensor);
	expect_tensor_near(sqrt_tensor, { 1.0f, 2.0f, 3.0f });

	vext::Tensor<float> square_tensor({ -2.0f, 3.0f });
	vext::unary<vext::Op::SQUARE>(square_tensor, vext::values({}), square_tensor);
	expect_tensor_near(square_tensor, { 4.0f, 9.0f });

	vext::Tensor<float> round_tensor({ 1.2f, 1.5f, -1.6f });
	vext::unary<vext::Op::ROUND>(round_tensor, vext::values({}), round_tensor);
	expect_tensor_near(round_tensor, { 1.0f, 2.0f, -2.0f });
}

TEST(TensorCpu, UnaryReturnedOutputLeavesInputUnchanged)
{
	const vext::Tensor<float> values({ -2.0f, 0.0f, 3.0f });
	const vext::Tensor<float> result = vext::unary<vext::Op::ABS>(values, vext::values({}));

	expect_tensor_near(result, { 2.0f, 0.0f, 3.0f });
	expect_tensor_near(values, { -2.0f, 0.0f, 3.0f });
}

TEST(TensorCpu, ActivationUnaryOpsMutateTensor)
{
	vext::Tensor<float> sigmoid_tensor({ 0.0f, 2.0f });
	vext::unary<vext::Op::SIGMOID>(sigmoid_tensor, vext::values({}), sigmoid_tensor);
	expect_tensor_near(sigmoid_tensor, { 0.5f, 1.0f / (1.0f + std::exp(-2.0f)) });

	vext::Tensor<float> soft_relu_tensor({ 0.0f, 2.0f });
	vext::unary<vext::Op::SOFT_RELU>(soft_relu_tensor, vext::values({}), soft_relu_tensor);
	expect_tensor_near(soft_relu_tensor, { std::log(2.0f), std::log(1.0f + std::exp(2.0f)) });

	vext::Tensor<float> relu_tensor({ -2.0f, 0.0f, 3.0f });
	vext::unary<vext::Op::RELU>(relu_tensor, vext::values({}), relu_tensor);
	expect_tensor_near(relu_tensor, { 0.0f, 0.0f, 3.0f });

	vext::Tensor<float> leaky_relu_tensor({ -2.0f, 3.0f });
	vext::unary<vext::Op::LEAKY_RELU>(leaky_relu_tensor, vext::values({ 0.25f }), leaky_relu_tensor);
	expect_tensor_near(leaky_relu_tensor, { -0.5f, 3.0f });

	vext::Tensor<float> elu_tensor({ -1.0f, 2.0f });
	vext::unary<vext::Op::ELU>(elu_tensor, vext::values({ 2.0f }), elu_tensor);
	expect_tensor_near(elu_tensor, { 2.0f * (std::exp(-1.0f) - 1.0f), 2.0f });

	vext::Tensor<float> swish_tensor({ -1.0f, 2.0f });
	vext::unary<vext::Op::SWISH>(swish_tensor, vext::values({ 1.0f }), swish_tensor);
	expect_tensor_near(swish_tensor, { -1.0f / (1.0f + std::exp(1.0f)), 2.0f / (1.0f + std::exp(-2.0f)) });
}

TEST(TensorCpu, NormalizationUnaryOpsMutateTensor)
{
	vext::Tensor<float> softmax_tensor({ 1.0f, 2.0f, 3.0f });
	vext::unary<vext::Op::SOFTMAX>(softmax_tensor, vext::values({}), softmax_tensor);
	const float softmax_sum = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);
	expect_tensor_near(softmax_tensor, { std::exp(1.0f) / softmax_sum, std::exp(2.0f) / softmax_sum, std::exp(3.0f) / softmax_sum });

	vext::Tensor<float> softmin_tensor({ 1.0f, 2.0f, 3.0f });
	vext::unary<vext::Op::SOFTMIN>(softmin_tensor, vext::values({}), softmin_tensor);
	const float softmin_sum = std::exp(-1.0f) + std::exp(-2.0f) + std::exp(-3.0f);
	expect_tensor_near(softmin_tensor, { std::exp(-1.0f) / softmin_sum, std::exp(-2.0f) / softmin_sum, std::exp(-3.0f) / softmin_sum });

	vext::Tensor<float> log_softmax_tensor({ 1.0f, 2.0f, 3.0f });
	vext::unary<vext::Op::LOGSOFTMAX>(log_softmax_tensor, vext::values({}), log_softmax_tensor);
	expect_tensor_near(log_softmax_tensor, { std::log(std::exp(1.0f) / softmax_sum), std::log(std::exp(2.0f) / softmax_sum), std::log(std::exp(3.0f) / softmax_sum) });
}

TEST(TensorCpu, ParameterizedUnaryOpsMutateTensor)
{
	vext::Tensor<float> linear_tensor({ -1.0f, 2.0f });
	vext::unary<vext::Op::LINEAR>(linear_tensor, vext::values({ 2.0f, 3.0f }), linear_tensor);
	expect_tensor_near(linear_tensor, { 1.0f, 7.0f });

	vext::Tensor<float> clip_tensor({ -2.0f, 0.5f, 3.0f });
	vext::unary<vext::Op::CLIP>(clip_tensor, vext::values({ -1.0f, 1.0f }), clip_tensor);
	expect_tensor_near(clip_tensor, { -1.0f, 0.5f, 1.0f });

	vext::Tensor<float> pow_tensor({ 2.0f, 3.0f });
	vext::unary<vext::Op::POW>(pow_tensor, vext::values({ 2.0f, 3.0f }), pow_tensor);
	expect_tensor_near(pow_tensor, { 16.0f, 54.0f });
}

TEST(TensorCpu, UnaryMinusNegatesInPlace)
{
	vext::Tensor<std::int32_t> tensor({ 1, -2, 3 });
	vext::unary<vext::Op::NEG>(tensor, vext::values({}), tensor);

	expect_tensor_values(tensor, { -1, 2, -3 });
}

TEST(TensorCpu, SupportsWholeTensorReductions)
{
	const vext::Tensor<float> tensor({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

	const vext::Tensor<float> sum      = vext::reduction<vext::Op::SUM>(tensor);
	const vext::Tensor<float> product  = vext::reduction<vext::Op::PROD>(tensor);
	const vext::Tensor<float> minimum  = vext::reduction<vext::Op::MIN>(tensor);
	const vext::Tensor<float> maximum  = vext::reduction<vext::Op::MAX>(tensor);
	const vext::Tensor<float> mean     = vext::reduction<vext::Op::MEAN>(tensor);
	const vext::Tensor<float> variance = vext::reduction<vext::Op::VAR>(tensor);
	const vext::Tensor<float> stddev   = vext::reduction<vext::Op::STD>(tensor);

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

	const vext::Tensor<float> column_sum  = vext::reduction<vext::Op::SUM>(tensor, vext::axes({ 0 }));
	const vext::Tensor<float> row_sum     = vext::reduction<vext::Op::SUM>(tensor, vext::axes({ 1 }));
	const vext::Tensor<float> column_mean = vext::reduction<vext::Op::MEAN>(tensor, vext::axes({ 0 }));
	const vext::Tensor<float> row_mean    = vext::reduction<vext::Op::MEAN>(tensor, vext::axes({ 1 }));
	const vext::Tensor<float> column_min  = vext::reduction<vext::Op::MIN>(tensor, vext::axes({ 0 }));
	const vext::Tensor<float> row_max     = vext::reduction<vext::Op::MAX>(tensor, vext::axes({ 1 }));

	expect_shape_eq(column_sum.dims(), { 3 });
	expect_tensor_near(column_sum, { 5.0f, 7.0f, 9.0f });

	expect_shape_eq(row_sum.dims(), { 2 });
	expect_tensor_near(row_sum, { 6.0f, 15.0f });

	expect_tensor_near(column_mean, { 2.5f, 3.5f, 4.5f });
	expect_tensor_near(row_mean, { 2.0f, 5.0f });
	expect_tensor_near(column_min, { 1.0f, 2.0f, 3.0f });
	expect_tensor_near(row_max, { 3.0f, 6.0f });
}

TEST(TensorCpu, ReducingAllAxesReturnsOneElementTensor)
{
	const vext::Tensor<float> vector({ 1.0f, 2.0f, 3.0f, 4.0f });
	const vext::Tensor<float> matrix({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });

	vext::Tensor<float> vector_sum({ 1 });
	vext::reduction<vext::Op::SUM>(vector, vext::axes({ 0 }), vector_sum);
	const auto matrix_sum = vext::reduction<vext::Op::SUM>(matrix, vext::axes({ 0, 1 }));

	expect_shape_eq(vector_sum.dims(), { 1 });
	expect_tensor_near(vector_sum, { 10.0f });
	expect_shape_eq(matrix_sum.dims(), { 1 });
	expect_tensor_near(matrix_sum, { 10.0f });
}

TEST(TensorCpu, ReductionsSupportMultipleAxes)
{
	const vext::Tensor<float> tensor({ { { 1.0f, 2.0f }, { 3.0f, 4.0f } }, { { 5.0f, 6.0f }, { 7.0f, 8.0f } } });

	const vext::Tensor<float> result = vext::reduction<vext::Op::SUM>(tensor, vext::axes({ 1, 2 }));

	expect_shape_eq(result.dims(), { 2 });
	expect_tensor_near(result, { 10.0f, 26.0f });
}

TEST(TensorCpu, ReductionAxesCanOutliveTheirCreatingExpression)
{
	const vext::Tensor<float> tensor({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });
	const vext::Axes          row_axis = vext::axes({ 1 });

	const vext::Tensor<float> result = vext::reduction<vext::Op::SUM>(tensor, row_axis);

	expect_shape_eq(result.dims(), { 2 });
	expect_tensor_near(result, { 3.0f, 7.0f });
}

TEST(TensorCpu, ReductionsRejectInvalidAxes)
{
	const vext::Tensor<float> tensor({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });

	EXPECT_THROW((void)vext::reduction<vext::Op::SUM>(tensor, vext::axes({ 2 })), std::runtime_error);
	EXPECT_THROW((void)vext::reduction<vext::Op::SUM>(tensor, vext::axes({ 0, 0 })), std::runtime_error);
	EXPECT_THROW((void)vext::reduction<vext::Op::SUM>(tensor, vext::axes({ 0, 1, 2 })), std::runtime_error);
	EXPECT_THROW((void)vext::reduction<vext::Op::SUM>(tensor, vext::axes({ -1 })), std::runtime_error);
}

TEST(TensorCpu, CsrScatterAggregatesNeighborRows)
{
	const vext::Tensor<float>         src({ { 1.0f, 2.0f }, { 3.0f, 4.0f }, { -2.0f, 5.0f }, { 4.0f, -1.0f } });
	const vext::Tensor<std::uint32_t> head({ 0U, 2U, 4U, 4U, 4U });
	const vext::Tensor<std::uint32_t> tail({ 0U, 2U, 1U, 3U });

	const vext::Tensor<float> sum      = vext::csr_scatter<vext::Op::SUM>(src, head, tail);
	const vext::Tensor<float> mean     = vext::csr_scatter<vext::Op::MEAN>(src, head, tail);
	const vext::Tensor<float> maximum  = vext::csr_scatter<vext::Op::MAX>(src, head, tail);
	const vext::Tensor<float> variance = vext::csr_scatter<vext::Op::VAR>(src, head, tail);
	const vext::Tensor<float> stddev   = vext::csr_scatter<vext::Op::STD>(src, head, tail);

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

	const vext::Tensor<float> minimum = vext::csr_scatter<vext::Op::MIN>(src, head, tail);
	const vext::Tensor<float> product = vext::csr_scatter<vext::Op::PROD>(src, head, tail);

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

	const vext::Tensor<float> sum      = vext::csr_spmv<vext::Op::SUM>(values, head, tail, x);
	const vext::Tensor<float> mean     = vext::csr_spmv<vext::Op::MEAN>(values, head, tail, x);
	const vext::Tensor<float> minimum  = vext::csr_spmv<vext::Op::MIN>(values, head, tail, x);
	const vext::Tensor<float> maximum  = vext::csr_spmv<vext::Op::MAX>(values, head, tail, x);
	const vext::Tensor<float> product  = vext::csr_spmv<vext::Op::PROD>(values, head, tail, x);
	const vext::Tensor<float> variance = vext::csr_spmv<vext::Op::VAR>(values, head, tail, x);
	const vext::Tensor<float> stddev   = vext::csr_spmv<vext::Op::STD>(values, head, tail, x);

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

	const vext::Tensor<float> result = vext::matmul(lhs, rhs);

	expect_shape_eq(result.dims(), { 2, 2 });
	expect_tensor_near(result, { 58.0f, 64.0f, 139.0f, 154.0f });
}

TEST(TensorCpu, MatmulSupportsBatchedLeftHandTensor)
{
	const vext::Tensor<float> lhs({ { { 1.0f, 2.0f }, { 3.0f, 4.0f } }, { { 5.0f, 6.0f }, { 7.0f, 8.0f } } });
	const vext::Tensor<float> rhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

	const vext::Tensor<float> result = vext::matmul(lhs, rhs);

	expect_shape_eq(result.dims(), { 2, 2, 3 });
	expect_tensor_near(result, { 9.0f, 12.0f, 15.0f, 19.0f, 26.0f, 33.0f, 29.0f, 40.0f, 51.0f, 39.0f, 54.0f, 69.0f });
}

TEST(TensorCpu, MatmulRejectsIncompatibleShapes)
{
	const vext::Tensor<float> lhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float> rhs({ { 1.0f, 2.0f }, { 3.0f, 4.0f }, { 5.0f, 6.0f }, { 7.0f, 8.0f } });

	EXPECT_THROW((void)vext::matmul(lhs, rhs), std::runtime_error);
}

TEST(TensorCpuNoise, UnaryAndBinaryPerturbTheParameterizedOperand)
{
	constexpr std::uint32_t seed    = 17;
	constexpr std::uint32_t counter = 9;
	set_noise_descriptor(seed, counter);

	vext::Tensor<float> unary_values({ 1.0f, 2.0f });
	vext::unary<vext::Op::LINEAR, vext::EvaluationMode::PERTURBED>(unary_values, vext::values({ 2.0f, 1.0f }), unary_values);

	expect_tensor_near(unary_values, { 2.0f * (1.0f + noise_value(seed, counter, 0)) + 1.0f, 2.0f * (2.0f + noise_value(seed, counter, 1)) + 1.0f });

	const vext::Tensor<float> lhs({ { 10.0f, 20.0f, 30.0f }, { 40.0f, 50.0f, 60.0f } });
	const vext::Tensor<float> rhs({ 1.0f, 2.0f, 3.0f });
	const vext::Tensor<float> result = vext::binary<vext::Op::ADD, vext::EvaluationMode::PERTURBED>(lhs, rhs);

	expect_tensor_near(result, { 11.0f + noise_value(seed, counter, 0), 22.0f + noise_value(seed, counter, 1), 33.0f + noise_value(seed, counter, 2), 41.0f + noise_value(seed, counter, 0), 52.0f + noise_value(seed, counter, 1), 63.0f + noise_value(seed, counter, 2) });
}

TEST(TensorCpuNoise, ReductionPerturbsValuesBeforeAggregation)
{
	constexpr std::uint32_t seed    = 23;
	constexpr std::uint32_t counter = 4;

	set_noise_descriptor(seed, counter);

	const vext::Tensor<float> values({ 1.0f, 2.0f, 4.0f });
	const vext::Tensor<float> sum          = vext::reduction<vext::Op::SUM, vext::EvaluationMode::PERTURBED>(values);
	const float               expected_sum = 7.0f + noise_value(seed, counter, 0) + noise_value(seed, counter, 1) + noise_value(seed, counter, 2);

	expect_scalar_near(sum, expected_sum);

	const vext::Tensor<float> variance          = vext::reduction<vext::Op::VAR, vext::EvaluationMode::PERTURBED>(values);
	const float               v0                = 1.0f + noise_value(seed, counter, 0);
	const float               v1                = 2.0f + noise_value(seed, counter, 1);
	const float               v2                = 4.0f + noise_value(seed, counter, 2);
	const float               mean              = (v0 + v1 + v2) / 3.0f;
	const float               expected_variance = ((v0 - mean) * (v0 - mean) + (v1 - mean) * (v1 - mean) + (v2 - mean) * (v2 - mean)) / 3.0f;

	expect_scalar_near(variance, expected_variance);
}

TEST(TensorCpuNoise, SparseOperationsPerturbSourceValuesBeforeAggregation)
{
	constexpr std::uint32_t seed    = 31;
	constexpr std::uint32_t counter = 7;

	set_noise_descriptor(seed, counter);

	const vext::Tensor<std::uint32_t> head({ 0U, 2U, 3U });
	const vext::Tensor<std::uint32_t> tail({ 0U, 1U, 1U });
	const vext::Tensor<float>         src({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });
	const vext::Tensor<float>         scatter = vext::csr_scatter<vext::Op::SUM, vext::EvaluationMode::PERTURBED>(src, head, tail);

	expect_tensor_near(scatter, { 4.0f + noise_value(seed, counter, 0) + noise_value(seed, counter, 2), 6.0f + noise_value(seed, counter, 1) + noise_value(seed, counter, 3), 3.0f + noise_value(seed, counter, 2), 4.0f + noise_value(seed, counter, 3) });

	const vext::Tensor<float> weights({ 1.0f, 2.0f, 3.0f });
	const vext::Tensor<float> x({ 2.0f, 5.0f });
	const vext::Tensor<float> spmv = vext::csr_spmv<vext::Op::SUM, vext::EvaluationMode::PERTURBED>(weights, head, tail, x);

	expect_tensor_near(spmv, { (1.0f + noise_value(seed, counter, 0)) * 2.0f + (2.0f + noise_value(seed, counter, 1)) * 5.0f, (3.0f + noise_value(seed, counter, 2)) * 5.0f });
}

TEST(TensorCpuNoise, MatmulPerturbsRightHandMatrixElements)
{
	constexpr std::uint32_t seed    = 43;
	constexpr std::uint32_t counter = 2;

	set_noise_descriptor(seed, counter);

	const vext::Tensor<float> lhs({ { 2.0f, 3.0f } });
	const vext::Tensor<float> rhs({ { 1.0f, 4.0f }, { 5.0f, 7.0f } });
	const vext::Tensor<float> result = vext::matmul<vext::EvaluationMode::PERTURBED>(lhs, rhs);

	expect_tensor_near(result, { 2.0f * (1.0f + noise_value(seed, counter, 0)) + 3.0f * (5.0f + noise_value(seed, counter, 2)), 2.0f * (4.0f + noise_value(seed, counter, 1)) + 3.0f * (7.0f + noise_value(seed, counter, 3)) });
}
