#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <vector>

#include <cuda_runtime.h>

#include <vext/ops.hpp>

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

TEST(TensorCuda, ConstructsCopiesAndReadsBackHostInitializerData)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> tensor({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });

	expect_shape(tensor.dims(), { 2, 2 });
	expect_tensor_near(tensor, { 1.0f, 2.0f, 3.0f, 4.0f });
}

TEST(TensorCuda, ConstructsFromDimensionsWithZeroInitializedStorage)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> tensor(2, 3);

	expect_shape(tensor.dims(), { 2, 3 });
	expect_tensor_near(tensor, { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f });
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

TEST(TensorCuda, MoveConstructorTransfersStorageAndLeavesSourceUsableAsMovedFromObject)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA> source({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });

	const vext::Tensor<float, vext::Backend::CUDA> moved(std::move(source));

	expect_shape(moved.dims(), { 2, 2 });
	expect_tensor_near(moved, { 1.0f, 2.0f, 3.0f, 4.0f });
	expect_shape(source.dims(), { vext::core::MIN_LENGTH });
}
