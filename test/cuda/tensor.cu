#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <numbers>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include <vext/tensor.hpp>

namespace
{

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
expect_cuda_tensor_near(
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

TEST(TensorCuda, ConstructsCopiesAndReadsBackHostInitializerData)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> tensor({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });

	expect_shape(tensor.shape(), { 2, 2 });
	expect_cuda_tensor_near(tensor, { 1.0f, 2.0f, 3.0f, 4.0f });
}

TEST(TensorCuda, ConstructsFromDimensionsWithZeroInitializedStorage)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> tensor(2, 3);

	expect_shape(tensor.shape(), { 2, 3 });
	expect_cuda_tensor_near(tensor, { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f });
}

TEST(TensorCuda, InitializerListRejectsInconsistentShape)
{
	EXPECT_THROW((vext::Tensor<float, vext::Backend::CUDA>({ { 1.0f, 2.0f }, { 3.0f } })), std::invalid_argument);
}

TEST(TensorCuda, ItemReadsFlatAndMultidimensionalIndices)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> tensor({ { 1.5f, 2.0f, 3.0f }, { 4.0f, 5.0f, -4.25f } });

	EXPECT_EQ(tensor.item(0), 1.5f);
	EXPECT_EQ(tensor.item(5), -4.25f);
	EXPECT_EQ(tensor.item(0, 1), 2.0f);
	EXPECT_EQ(tensor.item(1, 2), -4.25f);

	EXPECT_THROW((void)tensor.item(0, 0, 0), std::runtime_error);
	EXPECT_THROW((void)tensor.item(2, 0), std::runtime_error);
	EXPECT_THROW((void)tensor.item(0, 3), std::runtime_error);
}

TEST(TensorCuda, CopyConstructorCreatesIndependentStorage)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA> original({ 1.0f, 2.0f, 3.0f });
	vext::Tensor<float, vext::Backend::CUDA> copy(original);

	original += vext::Tensor<float, vext::Backend::CUDA>({ 10.0f, 10.0f, 10.0f });

	expect_cuda_tensor_near(copy, { 1.0f, 2.0f, 3.0f });
	expect_cuda_tensor_near(original, { 11.0f, 12.0f, 13.0f });
}

TEST(TensorCuda, CopyAssignmentCreatesIndependentStorage)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA> source({ 7.0f, 9.0f });
	vext::Tensor<float, vext::Backend::CUDA> target(1);

	target = source;
	source *= vext::Tensor<float, vext::Backend::CUDA>({ 2.0f, 3.0f });

	expect_cuda_tensor_near(target, { 7.0f, 9.0f });
	expect_cuda_tensor_near(source, { 14.0f, 27.0f });
}

TEST(TensorCuda, MoveConstructorTransfersStorageAndLeavesSourceUsableAsMovedFromObject)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA> source({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });

	const vext::Tensor<float, vext::Backend::CUDA> moved(std::move(source));

	expect_shape(moved.shape(), { 2, 2 });
	expect_cuda_tensor_near(moved, { 1.0f, 2.0f, 3.0f, 4.0f });
	expect_shape(source.shape(), { vext::core::MIN_LENGTH });
}

TEST(TensorCuda, LogicalComparisonsProduceMaskTensors)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> lhs({ 1.0f, 2.0f, 3.0f, 4.0f });
	const vext::Tensor<float, vext::Backend::CUDA> rhs({ 1.0f, 0.0f, 3.0f, 5.0f });

	expect_cuda_tensor_near(lhs == rhs, { 1.0f, 0.0f, 1.0f, 0.0f });
	expect_cuda_tensor_near(lhs != rhs, { 0.0f, 1.0f, 0.0f, 1.0f });
	expect_cuda_tensor_near(lhs < rhs, { 0.0f, 0.0f, 0.0f, 1.0f });
	expect_cuda_tensor_near(lhs <= rhs, { 1.0f, 0.0f, 1.0f, 1.0f });
	expect_cuda_tensor_near(lhs > rhs, { 0.0f, 1.0f, 0.0f, 0.0f });
	expect_cuda_tensor_near(lhs >= rhs, { 1.0f, 1.0f, 1.0f, 0.0f });
}

TEST(TensorCuda, LogicalComparisonsRejectIncompatibleShapes)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> lhs({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });
	const vext::Tensor<float, vext::Backend::CUDA> rhs({ 1.0f, 2.0f });

	EXPECT_THROW((void)(lhs == rhs), std::runtime_error);
}

TEST(TensorCuda, SupportsElementwiseArithmetic)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> lhs({ 1.0f, 2.0f, 3.0f });
	const vext::Tensor<float, vext::Backend::CUDA> rhs({ 4.0f, 5.0f, 6.0f });

	expect_cuda_tensor_near(lhs + rhs, { 5.0f, 7.0f, 9.0f });
	expect_cuda_tensor_near(rhs - lhs, { 3.0f, 3.0f, 3.0f });
	expect_cuda_tensor_near(lhs * rhs, { 4.0f, 10.0f, 18.0f });
	expect_cuda_tensor_near(rhs / lhs, { 4.0f, 2.5f, 2.0f });
	expect_cuda_tensor_near(lhs ^ vext::Tensor<float, vext::Backend::CUDA>({ 2.0f, 3.0f, 2.0f }), { 1.0f, 8.0f, 9.0f });
}

TEST(TensorCuda, ElementwiseArithmeticUsesCommonType)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<std::int32_t, vext::Backend::CUDA> lhs({ 1, 2 });
	const vext::Tensor<float, vext::Backend::CUDA>        rhs({ 0.5f, 1.25f });

	const vext::Tensor<float, vext::Backend::CUDA> result = lhs + rhs;

	static_assert(std::is_same_v<decltype(result), const vext::Tensor<float, vext::Backend::CUDA>>);
	expect_cuda_tensor_near(result, { 1.5f, 3.25f });
}

TEST(TensorCuda, ElementwiseArithmeticBroadcastsRightHandTensor)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> matrix({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float, vext::Backend::CUDA> bias({ 10.0f, 20.0f, 30.0f });

	const vext::Tensor<float, vext::Backend::CUDA> result = matrix + bias;

	expect_shape(result.shape(), { 2, 3 });
	expect_cuda_tensor_near(result, { 11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f });
}

TEST(TensorCuda, ElementwiseArithmeticRejectsIncompatibleShapes)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> lhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float, vext::Backend::CUDA> rhs({ 1.0f, 2.0f, 3.0f, 4.0f });

	EXPECT_THROW((void)(lhs + rhs), std::runtime_error);
}

TEST(TensorCuda, InPlaceArithmeticMutatesLeftHandSide)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA>       lhs({ 10.0f, 20.0f, 30.0f });
	const vext::Tensor<float, vext::Backend::CUDA> rhs({ 2.0f, 4.0f, 5.0f });

	lhs += rhs;
	expect_cuda_tensor_near(lhs, { 12.0f, 24.0f, 35.0f });

	lhs -= rhs;
	expect_cuda_tensor_near(lhs, { 10.0f, 20.0f, 30.0f });

	lhs *= rhs;
	expect_cuda_tensor_near(lhs, { 20.0f, 80.0f, 150.0f });

	lhs /= rhs;
	expect_cuda_tensor_near(lhs, { 10.0f, 20.0f, 30.0f });

	lhs ^= rhs;
	expect_cuda_tensor_near(lhs, { 100.0f, 160000.0f, 24300000.0f });
}

TEST(TensorCuda, PreluMutatesTensorWithElementwiseSlope)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA>       values({ -2.0f, -1.0f, 0.0f, 3.0f });
	const vext::Tensor<float, vext::Backend::CUDA> slopes({ 0.25f, 0.5f, 0.75f, 1.0f });

	values.prelu(slopes);

	expect_cuda_tensor_near(values, { -0.5f, -0.5f, 0.0f, 3.0f });
}

TEST(TensorCuda, SupportsUnaryActivations)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA> values({ -1.0f, 0.0f, 2.0f });

	values.relu();
	expect_cuda_tensor_near(values, { 0.0f, 0.0f, 2.0f });
}

TEST(TensorCuda, ParameterlessUnaryOperationsMutateTensor)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA> abs_tensor({ -1.0f, 0.0f, 4.0f });
	abs_tensor.abs();
	expect_cuda_tensor_near(abs_tensor, { 1.0f, 0.0f, 4.0f });

	vext::Tensor<float, vext::Backend::CUDA> sin_tensor({ 0.0f, static_cast<float>(std::numbers::pi / 2.0) });
	sin_tensor.sin();
	expect_cuda_tensor_near(sin_tensor, { 0.0f, 1.0f });

	vext::Tensor<float, vext::Backend::CUDA> cos_tensor({ 0.0f, static_cast<float>(std::numbers::pi) });
	cos_tensor.cos();
	expect_cuda_tensor_near(cos_tensor, { 1.0f, -1.0f });

	vext::Tensor<float, vext::Backend::CUDA> exp_tensor({ 0.0f, 1.0f });
	exp_tensor.exp();
	expect_cuda_tensor_near(exp_tensor, { 1.0f, std::exp(1.0f) });

	vext::Tensor<float, vext::Backend::CUDA> log_tensor({ 1.0f, std::exp(2.0f) });
	log_tensor.log();
	expect_cuda_tensor_near(log_tensor, { 0.0f, 2.0f });

	vext::Tensor<float, vext::Backend::CUDA> sqrt_tensor({ 1.0f, 4.0f, 9.0f });
	sqrt_tensor.sqrt();
	expect_cuda_tensor_near(sqrt_tensor, { 1.0f, 2.0f, 3.0f });

	vext::Tensor<float, vext::Backend::CUDA> square_tensor({ -2.0f, 3.0f });
	square_tensor.square();
	expect_cuda_tensor_near(square_tensor, { 4.0f, 9.0f });

	vext::Tensor<float, vext::Backend::CUDA> round_tensor({ 1.2f, 1.5f, -1.6f });
	round_tensor.round();
	expect_cuda_tensor_near(round_tensor, { 1.0f, 2.0f, -2.0f });
}

TEST(TensorCuda, ActivationUnaryOperationsMutateTensor)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA> sigmoid_tensor({ 0.0f, 2.0f });
	sigmoid_tensor.sigmoid();
	expect_cuda_tensor_near(sigmoid_tensor, { 0.5f, 1.0f / (1.0f + std::exp(-2.0f)) });

	vext::Tensor<float, vext::Backend::CUDA> soft_relu_tensor({ 0.0f, 2.0f });
	soft_relu_tensor.soft_relu();
	expect_cuda_tensor_near(soft_relu_tensor, { std::log(2.0f), std::log(1.0f + std::exp(2.0f)) });

	vext::Tensor<float, vext::Backend::CUDA> relu_tensor({ -2.0f, 0.0f, 3.0f });
	relu_tensor.relu();
	expect_cuda_tensor_near(relu_tensor, { 0.0f, 0.0f, 3.0f });

	vext::Tensor<float, vext::Backend::CUDA> leaky_relu_tensor({ -2.0f, 3.0f });
	leaky_relu_tensor.leaky_relu(0.25f);
	expect_cuda_tensor_near(leaky_relu_tensor, { -0.5f, 3.0f });

	vext::Tensor<float, vext::Backend::CUDA> elu_tensor({ -1.0f, 2.0f });
	elu_tensor.elu(2.0f);
	expect_cuda_tensor_near(elu_tensor, { 2.0f * (std::exp(-1.0f) - 1.0f), 2.0f });

	vext::Tensor<float, vext::Backend::CUDA> swish_tensor({ -1.0f, 2.0f });
	swish_tensor.swish(1.0f);
	expect_cuda_tensor_near(swish_tensor, { -1.0f / (1.0f + std::exp(1.0f)), 2.0f / (1.0f + std::exp(-2.0f)) });
}

TEST(TensorCuda, NormalizationUnaryOperationsMutateTensor)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA> softmax_tensor({ 1.0f, 2.0f, 3.0f });
	softmax_tensor.softmax();
	const float softmax_sum = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);
	expect_cuda_tensor_near(softmax_tensor, { std::exp(1.0f) / softmax_sum, std::exp(2.0f) / softmax_sum, std::exp(3.0f) / softmax_sum });

	vext::Tensor<float, vext::Backend::CUDA> softmin_tensor({ 1.0f, 2.0f, 3.0f });
	softmin_tensor.softmin();
	const float softmin_sum = std::exp(-1.0f) + std::exp(-2.0f) + std::exp(-3.0f);
	expect_cuda_tensor_near(softmin_tensor, { std::exp(-1.0f) / softmin_sum, std::exp(-2.0f) / softmin_sum, std::exp(-3.0f) / softmin_sum });

	vext::Tensor<float, vext::Backend::CUDA> log_softmax_tensor({ 1.0f, 2.0f, 3.0f });
	log_softmax_tensor.log_softmax();
	expect_cuda_tensor_near(log_softmax_tensor, { std::log(std::exp(1.0f) / softmax_sum), std::log(std::exp(2.0f) / softmax_sum), std::log(std::exp(3.0f) / softmax_sum) });
}

TEST(TensorCuda, ParameterizedUnaryOperationsMutateTensor)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA> linear_tensor({ -1.0f, 2.0f });
	linear_tensor.linear(2.0f, 3.0f);
	expect_cuda_tensor_near(linear_tensor, { 1.0f, 7.0f });

	vext::Tensor<float, vext::Backend::CUDA> clip_tensor({ -2.0f, 0.5f, 3.0f });
	clip_tensor.clip(-1.0f, 1.0f);
	expect_cuda_tensor_near(clip_tensor, { -1.0f, 0.5f, 1.0f });

	vext::Tensor<float, vext::Backend::CUDA> pow_tensor({ 2.0f, 3.0f });
	pow_tensor.pow(2.0f, 3.0f);
	expect_cuda_tensor_near(pow_tensor, { 16.0f, 54.0f });
}

TEST(TensorCuda, UnaryMinusNegatesInPlace)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA> tensor({ 1.0f, -2.0f, 3.0f });

	-tensor;

	expect_cuda_tensor_near(tensor, { -1.0f, 2.0f, -3.0f });
}

TEST(TensorCuda, SupportsMatrixMultiplication)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> lhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float, vext::Backend::CUDA> rhs({ { 7.0f, 8.0f }, { 9.0f, 10.0f }, { 11.0f, 12.0f } });

	const vext::Tensor<float, vext::Backend::CUDA> result = lhs.matmul(rhs);

	expect_shape(result.shape(), { 2, 2 });
	expect_cuda_tensor_near(result, { 58.0f, 64.0f, 139.0f, 154.0f });
}

TEST(TensorCuda, SupportsWholeTensorReductions)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> tensor({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

	expect_shape(tensor.sum().shape(), { 1 });
	expect_cuda_tensor_near(tensor.sum(), { 21.0f });
	expect_cuda_tensor_near(tensor.prod(), { 720.0f });
	expect_cuda_tensor_near(tensor.min(), { 1.0f });
	expect_cuda_tensor_near(tensor.max(), { 6.0f });
	expect_cuda_tensor_near(tensor.mean(), { 3.5f });
	expect_cuda_tensor_near(tensor.var(), { 17.5f / 6.0f });
	expect_cuda_tensor_near(tensor.std(), { std::sqrt(17.5f / 6.0f) });
}

TEST(TensorCuda, SupportsAxisReductions)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> tensor({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

	const vext::Tensor<float, vext::Backend::CUDA> column_sum = tensor.sum(0);
	const vext::Tensor<float, vext::Backend::CUDA> row_sum    = tensor.sum(1);

	expect_shape(column_sum.shape(), { 3 });
	expect_cuda_tensor_near(column_sum, { 5.0f, 7.0f, 9.0f });

	expect_shape(row_sum.shape(), { 2 });
	expect_cuda_tensor_near(row_sum, { 6.0f, 15.0f });
	expect_cuda_tensor_near(tensor.mean(0), { 2.5f, 3.5f, 4.5f });
	expect_cuda_tensor_near(tensor.max(1), { 3.0f, 6.0f });
}

TEST(TensorCuda, ReductionsSupportMultipleAxes)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> tensor({ { { 1.0f, 2.0f }, { 3.0f, 4.0f } }, { { 5.0f, 6.0f }, { 7.0f, 8.0f } } });

	const vext::Tensor<float, vext::Backend::CUDA> result = tensor.sum(1, 2);

	expect_shape(result.shape(), { 2 });
	expect_cuda_tensor_near(result, { 10.0f, 26.0f });
}

TEST(TensorCuda, ReductionsRejectInvalidAxes)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> tensor({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });

	EXPECT_THROW((void)tensor.sum(2), std::runtime_error);
	EXPECT_THROW((void)tensor.sum(0, 0), std::runtime_error);
	EXPECT_THROW((void)tensor.sum(0, 1, 2), std::runtime_error);
	EXPECT_THROW((void)tensor.sum(-1), std::runtime_error);
}

TEST(TensorCuda, MatmulSupportsBatchedLeftHandTensor)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> lhs({ { { 1.0f, 2.0f }, { 3.0f, 4.0f } }, { { 5.0f, 6.0f }, { 7.0f, 8.0f } } });
	const vext::Tensor<float, vext::Backend::CUDA> rhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

	const vext::Tensor<float, vext::Backend::CUDA> result = lhs.matmul(rhs);

	expect_shape(result.shape(), { 2, 2, 3 });
	expect_cuda_tensor_near(result, { 9.0f, 12.0f, 15.0f, 19.0f, 26.0f, 33.0f, 29.0f, 40.0f, 51.0f, 39.0f, 54.0f, 69.0f });
}

TEST(TensorCuda, MatmulRejectsIncompatibleShapes)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> lhs({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float, vext::Backend::CUDA> rhs({ { 1.0f, 2.0f }, { 3.0f, 4.0f }, { 5.0f, 6.0f }, { 7.0f, 8.0f } });

	EXPECT_THROW((void)lhs.matmul(rhs), std::runtime_error);
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

	const vext::Tensor<float, vext::Backend::CUDA> sum      = src.csr_scatter_sum(head, tail);
	const vext::Tensor<float, vext::Backend::CUDA> mean     = src.csr_scatter_mean(head, tail);
	const vext::Tensor<float, vext::Backend::CUDA> maximum  = src.csr_scatter_max(head, tail);
	const vext::Tensor<float, vext::Backend::CUDA> variance = src.csr_scatter_var(head, tail);
	const vext::Tensor<float, vext::Backend::CUDA> stddev   = src.csr_scatter_std(head, tail);

	expect_shape(sum.shape(), { 4, 2 });
	expect_shape(mean.shape(), { 4, 2 });
	expect_shape(maximum.shape(), { 4, 2 });
	expect_shape(variance.shape(), { 4, 2 });
	expect_shape(stddev.shape(), { 4, 2 });

	expect_cuda_tensor_near(sum, { -1.0f, 7.0f, 7.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f });
	expect_cuda_tensor_near(mean, { -0.5f, 3.5f, 3.5f, 1.5f, 0.0f, 0.0f, 0.0f, 0.0f });
	expect_cuda_tensor_near(maximum, { 1.0f, 5.0f, 4.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f });
	expect_cuda_tensor_near(variance, { 2.25f, 2.25f, 0.25f, 6.25f, 0.0f, 0.0f, 0.0f, 0.0f });
	expect_cuda_tensor_near(stddev, { 1.5f, 1.5f, 0.5f, 2.5f, 0.0f, 0.0f, 0.0f, 0.0f });
}

TEST(TensorCuda, CsrScatterMinAndProdUseOperationIdentity)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA>         src({ { 1.0f, 2.0f }, { 3.0f, 4.0f }, { -2.0f, 5.0f }, { 4.0f, -1.0f } });
	const vext::Tensor<std::uint32_t, vext::Backend::CUDA> head({ 0U, 2U, 4U, 4U, 4U });
	const vext::Tensor<std::uint32_t, vext::Backend::CUDA> tail({ 0U, 2U, 1U, 3U });

	const vext::Tensor<float, vext::Backend::CUDA> minimum = src.csr_scatter_min(head, tail);
	const vext::Tensor<float, vext::Backend::CUDA> product = src.csr_scatter_prod(head, tail);

	expect_shape(minimum.shape(), { 4, 2 });
	expect_shape(product.shape(), { 4, 2 });

	expect_cuda_tensor_near(minimum, { -2.0f, 2.0f, 3.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f });
	expect_cuda_tensor_near(product, { -2.0f, 10.0f, 12.0f, -4.0f, 0.0f, 0.0f, 0.0f, 0.0f });
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

	const vext::Tensor<float, vext::Backend::CUDA> sum      = values.csr_spmv_sum(head, tail, x);
	const vext::Tensor<float, vext::Backend::CUDA> mean     = values.csr_spmv_mean(head, tail, x);
	const vext::Tensor<float, vext::Backend::CUDA> minimum  = values.csr_spmv_min(head, tail, x);
	const vext::Tensor<float, vext::Backend::CUDA> maximum  = values.csr_spmv_max(head, tail, x);
	const vext::Tensor<float, vext::Backend::CUDA> product  = values.csr_spmv_prod(head, tail, x);
	const vext::Tensor<float, vext::Backend::CUDA> variance = values.csr_spmv_var(head, tail, x);
	const vext::Tensor<float, vext::Backend::CUDA> stddev   = values.csr_spmv_std(head, tail, x);

	expect_shape(sum.shape(), { 3 });
	expect_shape(mean.shape(), { 3 });
	expect_shape(minimum.shape(), { 3 });
	expect_shape(maximum.shape(), { 3 });
	expect_shape(product.shape(), { 3 });
	expect_shape(variance.shape(), { 3 });
	expect_shape(stddev.shape(), { 3 });

	expect_cuda_tensor_near(sum, { 8.0f, 13.0f, -9.0f });
	expect_cuda_tensor_near(mean, { 4.0f, 6.5f, -4.5f });
	expect_cuda_tensor_near(minimum, { 2.0f, -3.0f, -5.0f });
	expect_cuda_tensor_near(maximum, { 6.0f, 16.0f, -4.0f });
	expect_cuda_tensor_near(product, { 12.0f, -48.0f, 20.0f });
	expect_cuda_tensor_near(variance, { 4.0f, 90.25f, 0.25f });
	expect_cuda_tensor_near(stddev, { 2.0f, 9.5f, 0.5f });
}
