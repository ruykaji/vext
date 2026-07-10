#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include <cuda_runtime.h>

#include <vext/tensor.hpp>

namespace
{

void
expect_shape(
	const vext::Shape&                         shape,
	const std::initializer_list<std::uint32_t> expected)
{
	EXPECT_EQ(shape.dims(), std::vector<std::uint32_t>(expected));
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
	ASSERT_EQ(tensor.shape().length(), expected.size());

	for(std::uint32_t i = 0, end = tensor.shape().length(); i < end; ++i)
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
