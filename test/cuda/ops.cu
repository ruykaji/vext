#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <numbers>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include <vext/ops.hpp>

namespace
{

float
noise_value(
	const std::uint32_t seed,
	const std::uint32_t counter,
	const std::uint32_t index)
{
	std::uint32_t hash = seed ^ (counter * 0x85ebca6bu) ^ (index * 0x9e3779b9u);
	hash ^= hash >> 16;
	hash *= 0x7feb352du;
	hash ^= hash >> 15;
	hash *= 0x846ca68bu;
	hash ^= hash >> 16;

	return static_cast<float>(hash >> 8) * (1.0f / 16777216.0f);
}

void
set_noise_descriptor(
	const std::uint32_t seed,
	const std::uint32_t counter)
{
	vext::core::cuda::NoiseDescriptor& descriptor = vext::core::cuda::sequentional_noise_descriptor();
	descriptor.seed                               = seed;
	descriptor.counter                            = counter;
	descriptor.direction                          = 1;
}

void
expect_shape(
	const std::vector<std::uint32_t>&          shape,
	const std::initializer_list<std::uint32_t> expected)
{
	EXPECT_EQ(shape, std::vector<std::uint32_t>(expected));
}

bool
has_cuda_device()
{
	std::int32_t count = 0;
	cudaError_t  err   = cudaGetDeviceCount(&count);

	if(err == cudaErrorNoDevice || count == 0)
		{
			return false;
		}

	EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
	return err == cudaSuccess;
}

template <typename Tp>
void
expect_tensor_near(
	const vext::Tensor<Tp, vext::Backend::CUDA>& tensor,
	const std::vector<float>&                    expected,
	const float                                  tolerance = 1e-5f)
{
	ASSERT_EQ(tensor.length(), expected.size());

	for(std::uint32_t i = 0, end = tensor.length(); i < end; ++i)
		{
			EXPECT_NEAR(static_cast<float>(tensor.item(i)), expected[i], tolerance) << "at flat index " << i;
		}
}

}

TEST(TensorCuda, LogicalComparisonsProduceMaskTensors)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> lhs({ 1.0f, 2.0f, 3.0f, 4.0f });
	const vext::Tensor<float, vext::Backend::CUDA> rhs({ 1.0f, 0.0f, 3.0f, 5.0f });

	expect_tensor_near(vext::logical<vext::Op::EQUAL>(lhs, rhs), { 1.0f, 0.0f, 1.0f, 0.0f });
	expect_tensor_near(vext::logical<vext::Op::NOT_EQUAL>(lhs, rhs), { 0.0f, 1.0f, 0.0f, 1.0f });
	expect_tensor_near(vext::logical<vext::Op::LESS>(lhs, rhs), { 0.0f, 0.0f, 0.0f, 1.0f });
	expect_tensor_near(vext::logical<vext::Op::LESS_EQUAL>(lhs, rhs), { 1.0f, 0.0f, 1.0f, 1.0f });
	expect_tensor_near(vext::logical<vext::Op::GREATER>(lhs, rhs), { 0.0f, 1.0f, 0.0, 0.0f });
	expect_tensor_near(vext::logical<vext::Op::GREATER_EQUAL>(lhs, rhs), { 1.0f, 1.0f, 1.0f, 0.0f });
}

TEST(TensorCuda, LogicalComparisonsRejectIncompatibleShapes)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> lhs({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });
	const vext::Tensor<float, vext::Backend::CUDA> rhs({ 1.0f, 2.0f });

	EXPECT_THROW((void)(vext::logical<vext::Op::EQUAL>(lhs, rhs)), std::runtime_error);
}

TEST(TensorCuda, OperationsRejectEmptyInputTensors)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA>               empty;
	vext::Tensor<std::uint32_t, vext::Backend::CUDA>       empty_indices;
	const vext::Tensor<float, vext::Backend::CUDA>         values({ 1.0f, 2.0f });
	const vext::Tensor<float, vext::Backend::CUDA>         matrix({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });
	const vext::Tensor<std::uint32_t, vext::Backend::CUDA> head({ 0U, 1U, 2U });
	const vext::Tensor<std::uint32_t, vext::Backend::CUDA> tail({ 0U, 1U });

	EXPECT_THROW(vext::unary<vext::Op::RELU>(empty), std::runtime_error);
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

TEST(TensorCuda, SupportsElementwiseArithmetic)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> lhs({ 8.0f, 12.0f, 20.0f });
	const vext::Tensor<float, vext::Backend::CUDA> rhs({ 2.0f, 3.0f, 4.0f });

	const vext::Tensor<float, vext::Backend::CUDA> sum        = vext::binary<vext::Op::ADD>(lhs, rhs);
	const vext::Tensor<float, vext::Backend::CUDA> difference = vext::binary<vext::Op::SUB>(lhs, rhs);
	const vext::Tensor<float, vext::Backend::CUDA> product    = vext::binary<vext::Op::MUL>(lhs, rhs);
	const vext::Tensor<float, vext::Backend::CUDA> quotient   = vext::binary<vext::Op::DIV>(lhs, rhs);
	const vext::Tensor<float, vext::Backend::CUDA> power      = vext::binary<vext::Op::POW>(rhs, rhs);

	expect_tensor_near(sum, { 10.0f, 15.0f, 24.0f });
	expect_tensor_near(difference, { 6.0f, 9.0f, 16.0f });
	expect_tensor_near(product, { 16.0f, 36.0f, 80.0f });
	expect_tensor_near(quotient, { 4.0f, 4.0f, 5.0f });
	expect_tensor_near(power, { 4.0f, 27.0f, 256.0f });
}

TEST(TensorCuda, ElementwiseArithmeticUsesCommonType)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<std::int32_t, vext::Backend::CUDA> lhs({ 1, 2 });
	const vext::Tensor<float, vext::Backend::CUDA>        rhs({ 0.5f, 1.25f });

	const vext::Tensor<float, vext::Backend::CUDA> result = vext::binary<vext::Op::ADD>(lhs, rhs);

	static_assert(std::is_same_v<decltype(result), const vext::Tensor<float, vext::Backend::CUDA>>);
	expect_tensor_near(result, { 1.5f, 3.25f });
}

TEST(TensorCuda, ElementwiseArithmeticBroadcastsRightHandTensor)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> matrix({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float, vext::Backend::CUDA> bias({ 10.0f, 20.0f, 30.0f });

	const vext::Tensor<float, vext::Backend::CUDA> result = vext::binary<vext::Op::ADD>(matrix, bias);

	expect_shape(result.dims(), { 2, 3 });
	expect_tensor_near(result, { 11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f });
}

TEST(TensorCuda, ElementwiseArithmeticRejectsIncompatibleShapes)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> lhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float, vext::Backend::CUDA> rhs({ 1.0f, 2.0f, 3.0f, 4.0f });

	EXPECT_THROW((void)(vext::binary<vext::Op::ADD>(lhs, rhs)), std::runtime_error);
}

TEST(TensorCuda, PreluMutatesTensorWithElementwiseSlope)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA>       values({ -2.0f, -1.0f, 0.0f, 3.0f });
	const vext::Tensor<float, vext::Backend::CUDA> slopes({ 0.25f, 0.5f, 0.75f, 1.0f });

	vext::binary<vext::Op::PRELU>(values, slopes, values);

	expect_tensor_near(values, { -0.5f, -0.5f, 0.0f, 3.0f });
}

TEST(TensorCuda, ParameterlessUnaryOpsMutateTensor)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA> abs_tensor({ -1.0f, 0.0f, 4.0f });
	vext::unary<vext::Op::ABS>(abs_tensor);
	expect_tensor_near(abs_tensor, { 1.0f, 0.0f, 4.0f });

	vext::Tensor<float, vext::Backend::CUDA> sin_tensor({ 0.0f, static_cast<float>(std::numbers::pi / 2.0) });
	vext::unary<vext::Op::SIN>(sin_tensor);
	expect_tensor_near(sin_tensor, { 0.0f, 1.0f });

	vext::Tensor<float, vext::Backend::CUDA> cos_tensor({ 0.0f, static_cast<float>(std::numbers::pi) });
	vext::unary<vext::Op::COS>(cos_tensor);
	expect_tensor_near(cos_tensor, { 1.0f, -1.0f });

	vext::Tensor<float, vext::Backend::CUDA> exp_tensor({ 0.0f, 1.0f });
	vext::unary<vext::Op::EXP>(exp_tensor);
	expect_tensor_near(exp_tensor, { 1.0f, std::exp(1.0f) });

	vext::Tensor<float, vext::Backend::CUDA> log_tensor({ 1.0f, std::exp(2.0f) });
	vext::unary<vext::Op::LOG>(log_tensor);
	expect_tensor_near(log_tensor, { 0.0f, 2.0f });

	vext::Tensor<float, vext::Backend::CUDA> sqrt_tensor({ 1.0f, 4.0f, 9.0f });
	vext::unary<vext::Op::SQRT>(sqrt_tensor);
	expect_tensor_near(sqrt_tensor, { 1.0f, 2.0f, 3.0f });

	vext::Tensor<float, vext::Backend::CUDA> square_tensor({ -2.0f, 3.0f });
	vext::unary<vext::Op::SQUARE>(square_tensor);
	expect_tensor_near(square_tensor, { 4.0f, 9.0f });

	vext::Tensor<float, vext::Backend::CUDA> round_tensor({ 1.2f, 1.5f, -1.6f });
	vext::unary<vext::Op::ROUND>(round_tensor);
	expect_tensor_near(round_tensor, { 1.0f, 2.0f, -2.0f });
}

TEST(TensorCuda, ActivationUnaryOpsMutateTensor)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA> sigmoid_tensor({ 0.0f, 2.0f });
	vext::unary<vext::Op::SIGMOID>(sigmoid_tensor);
	expect_tensor_near(sigmoid_tensor, { 0.5f, 1.0f / (1.0f + std::exp(-2.0f)) });

	vext::Tensor<float, vext::Backend::CUDA> soft_relu_tensor({ 0.0f, 2.0f });
	vext::unary<vext::Op::SOFT_RELU>(soft_relu_tensor);
	expect_tensor_near(soft_relu_tensor, { std::log(2.0f), std::log(1.0f + std::exp(2.0f)) });

	vext::Tensor<float, vext::Backend::CUDA> relu_tensor({ -2.0f, 0.0f, 3.0f });
	vext::unary<vext::Op::RELU>(relu_tensor);
	expect_tensor_near(relu_tensor, { 0.0f, 0.0f, 3.0f });

	vext::Tensor<float, vext::Backend::CUDA> leaky_relu_tensor({ -2.0f, 3.0f });
	vext::unary<vext::Op::LEAKY_RELU>(leaky_relu_tensor, 0.25f);
	expect_tensor_near(leaky_relu_tensor, { -0.5f, 3.0f });

	vext::Tensor<float, vext::Backend::CUDA> elu_tensor({ -1.0f, 2.0f });
	vext::unary<vext::Op::ELU>(elu_tensor, 2.0f);
	expect_tensor_near(elu_tensor, { 2.0f * (std::exp(-1.0f) - 1.0f), 2.0f });

	vext::Tensor<float, vext::Backend::CUDA> swish_tensor({ -1.0f, 2.0f });
	vext::unary<vext::Op::SWISH>(swish_tensor, 1.0f);
	expect_tensor_near(swish_tensor, { -1.0f / (1.0f + std::exp(1.0f)), 2.0f / (1.0f + std::exp(-2.0f)) });
}

TEST(TensorCuda, NormalizationUnaryOpsMutateTensor)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA> softmax_tensor({ 1.0f, 2.0f, 3.0f });
	vext::unary<vext::Op::SOFTMAX>(softmax_tensor);
	const float softmax_sum = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);
	expect_tensor_near(softmax_tensor, { std::exp(1.0f) / softmax_sum, std::exp(2.0f) / softmax_sum, std::exp(3.0f) / softmax_sum });

	vext::Tensor<float, vext::Backend::CUDA> softmin_tensor({ 1.0f, 2.0f, 3.0f });
	vext::unary<vext::Op::SOFTMIN>(softmin_tensor);
	const float softmin_sum = std::exp(-1.0f) + std::exp(-2.0f) + std::exp(-3.0f);
	expect_tensor_near(softmin_tensor, { std::exp(-1.0f) / softmin_sum, std::exp(-2.0f) / softmin_sum, std::exp(-3.0f) / softmin_sum });

	vext::Tensor<float, vext::Backend::CUDA> log_softmax_tensor({ 1.0f, 2.0f, 3.0f });
	vext::unary<vext::Op::LOGSOFTMAX>(log_softmax_tensor);
	expect_tensor_near(log_softmax_tensor, { std::log(std::exp(1.0f) / softmax_sum), std::log(std::exp(2.0f) / softmax_sum), std::log(std::exp(3.0f) / softmax_sum) });
}

TEST(TensorCuda, ParameterizedUnaryOpsMutateTensor)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA> linear_tensor({ -1.0f, 2.0f });
	vext::unary<vext::Op::LINEAR>(linear_tensor, 2.0f, 3.0f);
	expect_tensor_near(linear_tensor, { 1.0f, 7.0f });

	vext::Tensor<float, vext::Backend::CUDA> clip_tensor({ -2.0f, 0.5f, 3.0f });
	vext::unary<vext::Op::CLIP>(clip_tensor, -1.0f, 1.0f);
	expect_tensor_near(clip_tensor, { -1.0f, 0.5f, 1.0f });

	vext::Tensor<float, vext::Backend::CUDA> pow_tensor({ 2.0f, 3.0f });
	vext::unary<vext::Op::POW>(pow_tensor, 2.0f, 3.0f);
	expect_tensor_near(pow_tensor, { 16.0f, 54.0f });
}

TEST(TensorCuda, UnaryMinusNegatesInPlace)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA> tensor({ 1.0f, -2.0f, 3.0f });
	vext::unary<vext::Op::NEG>(tensor);

	expect_tensor_near(tensor, { -1.0f, 2.0f, -3.0f });
}

TEST(TensorCuda, SupportsWholeTensorReductions)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> tensor({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

	const vext::Tensor<float, vext::Backend::CUDA> sum      = vext::reduction<vext::Op::SUM>(tensor);
	const vext::Tensor<float, vext::Backend::CUDA> product  = vext::reduction<vext::Op::PROD>(tensor);
	const vext::Tensor<float, vext::Backend::CUDA> minimum  = vext::reduction<vext::Op::MIN>(tensor);
	const vext::Tensor<float, vext::Backend::CUDA> maximum  = vext::reduction<vext::Op::MAX>(tensor);
	const vext::Tensor<float, vext::Backend::CUDA> mean     = vext::reduction<vext::Op::MEAN>(tensor);
	const vext::Tensor<float, vext::Backend::CUDA> variance = vext::reduction<vext::Op::VAR>(tensor);
	const vext::Tensor<float, vext::Backend::CUDA> stddev   = vext::reduction<vext::Op::STD>(tensor);

	expect_tensor_near(sum, { 21.0f });
	expect_tensor_near(product, { 720.0f });
	expect_tensor_near(minimum, { 1.0f });
	expect_tensor_near(maximum, { 6.0f });
	expect_tensor_near(mean, { 3.5f });
	expect_tensor_near(variance, { 17.5f / 6.0f });
	expect_tensor_near(stddev, { std::sqrt(17.5f / 6.0f) });
}

TEST(TensorCuda, ReductionsSupportSingleAxis)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> tensor({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

	const vext::Tensor<float, vext::Backend::CUDA> column_sum  = vext::reduction<vext::Op::SUM>(tensor, vext::axes({ 0 }));
	const vext::Tensor<float, vext::Backend::CUDA> row_sum     = vext::reduction<vext::Op::SUM>(tensor, vext::axes({ 1 }));
	const vext::Tensor<float, vext::Backend::CUDA> column_mean = vext::reduction<vext::Op::MEAN>(tensor, vext::axes({ 0 }));
	const vext::Tensor<float, vext::Backend::CUDA> row_mean    = vext::reduction<vext::Op::MEAN>(tensor, vext::axes({ 1 }));
	const vext::Tensor<float, vext::Backend::CUDA> column_min  = vext::reduction<vext::Op::MIN>(tensor, vext::axes({ 0 }));
	const vext::Tensor<float, vext::Backend::CUDA> row_max     = vext::reduction<vext::Op::MAX>(tensor, vext::axes({ 1 }));

	expect_shape(column_sum.dims(), { 3 });
	expect_tensor_near(column_sum, { 5.0f, 7.0f, 9.0f });

	expect_shape(row_sum.dims(), { 2 });
	expect_tensor_near(row_sum, { 6.0f, 15.0f });

	expect_tensor_near(column_mean, { 2.5f, 3.5f, 4.5f });
	expect_tensor_near(row_mean, { 2.0f, 5.0f });
	expect_tensor_near(column_min, { 1.0f, 2.0f, 3.0f });
	expect_tensor_near(row_max, { 3.0f, 6.0f });
}

TEST(TensorCuda, ReducingAllAxesReturnsOneElementTensor)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> vector({ 1.0f, 2.0f, 3.0f, 4.0f });
	const vext::Tensor<float, vext::Backend::CUDA> matrix({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });

	vext::Tensor<float, vext::Backend::CUDA> vector_sum({ 1 });
	vext::reduction<vext::Op::SUM>(vector, vext::axes({ 0 }), vector_sum);
	const auto matrix_sum = vext::reduction<vext::Op::SUM>(matrix, vext::axes({ 0, 1 }));

	expect_shape(vector_sum.dims(), { 1 });
	expect_tensor_near(vector_sum, { 10.0f });
	expect_shape(matrix_sum.dims(), { 1 });
	expect_tensor_near(matrix_sum, { 10.0f });
}

TEST(TensorCuda, ReductionsSupportMultipleAxes)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> tensor({ { { 1.0f, 2.0f }, { 3.0f, 4.0f } }, { { 5.0f, 6.0f }, { 7.0f, 8.0f } } });

	const vext::Tensor<float, vext::Backend::CUDA> result = vext::reduction<vext::Op::SUM>(tensor, vext::axes({ 1, 2 }));

	expect_shape(result.dims(), { 2 });
	expect_tensor_near(result, { 10.0f, 26.0f });
}

TEST(TensorCuda, ReductionsRejectInvalidAxes)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> tensor({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });

	EXPECT_THROW((void)vext::reduction<vext::Op::SUM>(tensor, vext::axes({ 2 })), std::runtime_error);
	EXPECT_THROW((void)vext::reduction<vext::Op::SUM>(tensor, vext::axes({ 0, 0 })), std::runtime_error);
	EXPECT_THROW((void)vext::reduction<vext::Op::SUM>(tensor, vext::axes({ 0, 1, 2 })), std::runtime_error);
	EXPECT_THROW((void)vext::reduction<vext::Op::SUM>(tensor, vext::axes({ -1 })), std::runtime_error);
}

TEST(TensorCuda, CsrScatterAggregatesNeighborRows)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA>         src({ { 1.0f, 2.0f }, { 3.0f, 4.0f }, { -2.0f, 5.0f }, { 4.0f, -1.0f } });
	const vext::Tensor<std::uint32_t, vext::Backend::CUDA> head({ 0U, 2U, 4U, 4U, 4U });
	const vext::Tensor<std::uint32_t, vext::Backend::CUDA> tail({ 0U, 2U, 1U, 3U });

	const vext::Tensor<float, vext::Backend::CUDA> sum      = vext::csr_scatter<vext::Op::SUM>(src, head, tail);
	const vext::Tensor<float, vext::Backend::CUDA> mean     = vext::csr_scatter<vext::Op::MEAN>(src, head, tail);
	const vext::Tensor<float, vext::Backend::CUDA> maximum  = vext::csr_scatter<vext::Op::MAX>(src, head, tail);
	const vext::Tensor<float, vext::Backend::CUDA> variance = vext::csr_scatter<vext::Op::VAR>(src, head, tail);
	const vext::Tensor<float, vext::Backend::CUDA> stddev   = vext::csr_scatter<vext::Op::STD>(src, head, tail);

	expect_shape(sum.dims(), { 4, 2 });
	expect_shape(mean.dims(), { 4, 2 });
	expect_shape(maximum.dims(), { 4, 2 });
	expect_shape(variance.dims(), { 4, 2 });
	expect_shape(stddev.dims(), { 4, 2 });

	expect_tensor_near(sum, { -1.0f, 7.0f, 7.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f });
	expect_tensor_near(mean, { -0.5f, 3.5f, 3.5f, 1.5f, 0.0f, 0.0f, 0.0f, 0.0f });
	expect_tensor_near(maximum, { 1.0f, 5.0f, 4.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f });
	expect_tensor_near(variance, { 2.25f, 2.25f, 0.25f, 6.25f, 0.0f, 0.0f, 0.0f, 0.0f });
	expect_tensor_near(stddev, { 1.5f, 1.5f, 0.5f, 2.5f, 0.0f, 0.0f, 0.0f, 0.0f });
}

TEST(TensorCuda, CsrScatterMinAndProdUseOpIdentity)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA>         src({ { 1.0f, 2.0f }, { 3.0f, 4.0f }, { -2.0f, 5.0f }, { 4.0f, -1.0f } });
	const vext::Tensor<std::uint32_t, vext::Backend::CUDA> head({ 0U, 2U, 4U, 4U, 4U });
	const vext::Tensor<std::uint32_t, vext::Backend::CUDA> tail({ 0U, 2U, 1U, 3U });

	const vext::Tensor<float, vext::Backend::CUDA> minimum = vext::csr_scatter<vext::Op::MIN>(src, head, tail);
	const vext::Tensor<float, vext::Backend::CUDA> product = vext::csr_scatter<vext::Op::PROD>(src, head, tail);

	expect_shape(minimum.dims(), { 4, 2 });
	expect_shape(product.dims(), { 4, 2 });

	expect_tensor_near(minimum, { -2.0f, 2.0f, 3.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f });
	expect_tensor_near(product, { -2.0f, 10.0f, 12.0f, -4.0f, 0.0f, 0.0f, 0.0f, 0.0f });
}

TEST(TensorCuda, CsrSpmvAggregatesSparseMatrixVectorProducts)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA>         values({ 1.0f, 2.0f, 3.0f, 4.0f, -2.0f, 5.0f });
	const vext::Tensor<std::uint32_t, vext::Backend::CUDA> head({ 0U, 2U, 4U, 6U });
	const vext::Tensor<std::uint32_t, vext::Backend::CUDA> tail({ 0U, 2U, 1U, 3U, 0U, 1U });
	const vext::Tensor<float, vext::Backend::CUDA>         x({ 2.0f, -1.0f, 3.0f, 4.0f });

	const vext::Tensor<float, vext::Backend::CUDA> sum      = vext::csr_spmv<vext::Op::SUM>(values, head, tail, x);
	const vext::Tensor<float, vext::Backend::CUDA> mean     = vext::csr_spmv<vext::Op::MEAN>(values, head, tail, x);
	const vext::Tensor<float, vext::Backend::CUDA> minimum  = vext::csr_spmv<vext::Op::MIN>(values, head, tail, x);
	const vext::Tensor<float, vext::Backend::CUDA> maximum  = vext::csr_spmv<vext::Op::MAX>(values, head, tail, x);
	const vext::Tensor<float, vext::Backend::CUDA> product  = vext::csr_spmv<vext::Op::PROD>(values, head, tail, x);
	const vext::Tensor<float, vext::Backend::CUDA> variance = vext::csr_spmv<vext::Op::VAR>(values, head, tail, x);
	const vext::Tensor<float, vext::Backend::CUDA> stddev   = vext::csr_spmv<vext::Op::STD>(values, head, tail, x);

	expect_shape(sum.dims(), { 3 });
	expect_shape(mean.dims(), { 3 });
	expect_shape(minimum.dims(), { 3 });
	expect_shape(maximum.dims(), { 3 });
	expect_shape(product.dims(), { 3 });
	expect_shape(variance.dims(), { 3 });
	expect_shape(stddev.dims(), { 3 });

	expect_tensor_near(sum, { 8.0f, 13.0f, -9.0f });
	expect_tensor_near(mean, { 4.0f, 6.5f, -4.5f });
	expect_tensor_near(minimum, { 2.0f, -3.0f, -5.0f });
	expect_tensor_near(maximum, { 6.0f, 16.0f, -4.0f });
	expect_tensor_near(product, { 12.0f, -48.0f, 20.0f });
	expect_tensor_near(variance, { 4.0f, 90.25f, 0.25f });
	expect_tensor_near(stddev, { 2.0f, 9.5f, 0.5f });
}

TEST(TensorCuda, SupportsMatrixMultiplication)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> lhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float, vext::Backend::CUDA> rhs({ { 7.0f, 8.0f }, { 9.0f, 10.0f }, { 11.0f, 12.0f } });

	const vext::Tensor<float, vext::Backend::CUDA> result = vext::matmul(lhs, rhs);

	expect_shape(result.dims(), { 2, 2 });
	expect_tensor_near(result, { 58.0f, 64.0f, 139.0f, 154.0f });
}

TEST(TensorCuda, MatmulSupportsBatchedLeftHandTensor)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> lhs({ { { 1.0f, 2.0f }, { 3.0f, 4.0f } }, { { 5.0f, 6.0f }, { 7.0f, 8.0f } } });
	const vext::Tensor<float, vext::Backend::CUDA> rhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

	const vext::Tensor<float, vext::Backend::CUDA> result = vext::matmul(lhs, rhs);

	expect_shape(result.dims(), { 2, 2, 3 });
	expect_tensor_near(result, { 9.0f, 12.0f, 15.0f, 19.0f, 26.0f, 33.0f, 29.0f, 40.0f, 51.0f, 39.0f, 54.0f, 69.0f });
}

TEST(TensorCuda, MatmulRejectsIncompatibleShapes)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> lhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float, vext::Backend::CUDA> rhs({ { 1.0f, 2.0f }, { 3.0f, 4.0f }, { 5.0f, 6.0f }, { 7.0f, 8.0f } });

	EXPECT_THROW((void)vext::matmul(lhs, rhs), std::runtime_error);
}

TEST(TensorCudaNoise, UnaryBinaryAndReductionUseDeterministicNoise)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	constexpr std::uint32_t seed    = 17;
	constexpr std::uint32_t counter = 9;
	set_noise_descriptor(seed, counter);

	vext::Tensor<float, vext::Backend::CUDA> unary_values({ 1.0f, 2.0f });
	vext::unary<vext::Op::LINEAR, vext::ParameterMode::PERTURBED>(unary_values, 2.0f, 1.0f);
	expect_tensor_near(unary_values, { 2.0f * (1.0f + noise_value(seed, counter, 0)) + 1.0f, 2.0f * (2.0f + noise_value(seed, counter, 1)) + 1.0f });

	const vext::Tensor<float, vext::Backend::CUDA> lhs({ { 10.0f, 20.0f, 30.0f }, { 40.0f, 50.0f, 60.0f } });
	const vext::Tensor<float, vext::Backend::CUDA> rhs({ 1.0f, 2.0f, 3.0f });
	const auto                                     result = vext::binary<vext::Op::ADD, vext::ParameterMode::PERTURBED>(lhs, rhs);
	expect_tensor_near(result, { 11.0f + noise_value(seed, counter, 0), 22.0f + noise_value(seed, counter, 1), 33.0f + noise_value(seed, counter, 2), 41.0f + noise_value(seed, counter, 0), 52.0f + noise_value(seed, counter, 1), 63.0f + noise_value(seed, counter, 2) });

	const vext::Tensor<float, vext::Backend::CUDA> values({ 1.0f, 2.0f, 4.0f });
	const auto                                     sum = vext::reduction<vext::Op::SUM, vext::ParameterMode::PERTURBED>(values);
	expect_tensor_near(sum, { 7.0f + noise_value(seed, counter, 0) + noise_value(seed, counter, 1) + noise_value(seed, counter, 2) });
}

TEST(TensorCudaNoise, SparseOperationsAndMatmulPerturbTheirParameterizedInputs)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	constexpr std::uint32_t seed    = 31;
	constexpr std::uint32_t counter = 7;
	set_noise_descriptor(seed, counter);

	const vext::Tensor<std::uint32_t, vext::Backend::CUDA> head({ 0U, 2U, 3U });
	const vext::Tensor<std::uint32_t, vext::Backend::CUDA> tail({ 0U, 1U, 1U });
	const vext::Tensor<float, vext::Backend::CUDA>         src({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });
	const auto                                             scatter = vext::csr_scatter<vext::Op::SUM, vext::ParameterMode::PERTURBED>(src, head, tail);
	expect_tensor_near(scatter, { 4.0f + noise_value(seed, counter, 0) + noise_value(seed, counter, 2), 6.0f + noise_value(seed, counter, 1) + noise_value(seed, counter, 3), 3.0f + noise_value(seed, counter, 2), 4.0f + noise_value(seed, counter, 3) });

	const vext::Tensor<float, vext::Backend::CUDA> weights({ 1.0f, 2.0f, 3.0f });
	const vext::Tensor<float, vext::Backend::CUDA> x({ 2.0f, 5.0f });
	const auto                                     spmv = vext::csr_spmv<vext::Op::SUM, vext::ParameterMode::PERTURBED>(weights, head, tail, x);
	expect_tensor_near(spmv, { (1.0f + noise_value(seed, counter, 0)) * 2.0f + (2.0f + noise_value(seed, counter, 1)) * 5.0f, (3.0f + noise_value(seed, counter, 2)) * 5.0f });

	const vext::Tensor<float, vext::Backend::CUDA> mat_lhs({ { 2.0f, 3.0f } });
	const vext::Tensor<float, vext::Backend::CUDA> mat_rhs({ { 1.0f, 4.0f }, { 5.0f, 7.0f } });
	const auto                                     product = vext::matmul<vext::ParameterMode::PERTURBED>(mat_lhs, mat_rhs);
	expect_tensor_near(product, { 2.0f * (1.0f + noise_value(seed, counter, 0)) + 3.0f * (5.0f + noise_value(seed, counter, 2)), 2.0f * (4.0f + noise_value(seed, counter, 1)) + 3.0f * (7.0f + noise_value(seed, counter, 3)) });
}
