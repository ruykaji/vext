#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include <vext/core/cuda/allocator.cuh>
#include <vext/core/cuda/operations/elementwise_binary.cuh>
#include <vext/core/cuda/operations/elementwise_logical.cuh>
#include <vext/core/cuda/operations/elementwise_unary.cuh>
#include <vext/core/cuda/operations/linear_algebra.cuh>
#include <vext/core/cuda/operations/memory.cuh>
#include <vext/core/cuda/operations/reduction.cuh>

namespace
{

void
check_cuda(
	const cudaError_t err)
{
	ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
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
std::vector<Tp>
copy_to_host(
	const Tp*           device,
	const std::uint32_t size)
{
	std::vector<Tp> host(size);
	check_cuda(cudaMemcpy(host.data(), device, size * sizeof(Tp), cudaMemcpyDeviceToHost));
	return host;
}

template <typename Tp>
Tp*
copy_to_device(
	const std::vector<Tp>& host)
{
	Tp* device = vext::core::cuda::allocator::allocate<Tp>(host.size());
	vext::core::cuda::operations::memcpy<Tp, vext::Backend::CUDA, vext::Backend::CPU>(device, host.data(), host.size());
	return device;
}

void
expect_near(
	const std::vector<float>& actual,
	const std::vector<float>& expected,
	const float               tolerance = 1e-5f)
{
	ASSERT_EQ(actual.size(), expected.size());

	for(std::uint64_t i = 0; i < actual.size(); ++i)
		{
			EXPECT_NEAR(actual[i], expected[i], tolerance) << "at index " << i;
		}
}

}

TEST(CoreCudaAllocator, AllocatesZeroInitializedDeviceMemory)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	float* ptr = vext::core::cuda::allocator::allocate<float>(4);

	expect_near(copy_to_host(ptr, 4), { 0.0f, 0.0f, 0.0f, 0.0f });

	vext::core::cuda::allocator::deallocate(ptr);
	vext::core::cuda::allocator::free();
}

TEST(CoreCudaMemory, CopiesBetweenHostAndDevice)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const std::vector<float> host{ 1.0f, 2.0f, 3.0f };
	float*                   device = copy_to_device(host);

	expect_near(copy_to_host(device, 3), host);

	vext::core::cuda::allocator::deallocate(device);
	vext::core::cuda::allocator::free();
}

TEST(CoreCudaBinaryOperations, ComputesElementwiseAdd)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	float* a   = copy_to_device(std::vector<float>{ 1.0f, 2.0f, 3.0f });
	float* b   = copy_to_device(std::vector<float>{ 4.0f, 5.0f, 6.0f });
	float* out = vext::core::cuda::allocator::allocate<float>(3);

	vext::core::cuda::operations::binary<vext::core::BinaryOperation::ADD>(out, a, b, 3);
	check_cuda(cudaDeviceSynchronize());

	expect_near(copy_to_host(out, 3), { 5.0f, 7.0f, 9.0f });

	vext::core::cuda::allocator::deallocate(a);
	vext::core::cuda::allocator::deallocate(b);
	vext::core::cuda::allocator::deallocate(out);
	vext::core::cuda::allocator::free();
}

TEST(CoreCudaUnaryOperations, ComputesRelu)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	float* values = copy_to_device(std::vector<float>{ -1.0f, 0.0f, 2.0f });

	vext::core::cuda::operations::unary<vext::core::UnaryOperation::RELU>(values, 3);
	check_cuda(cudaDeviceSynchronize());

	expect_near(copy_to_host(values, 3), { 0.0f, 0.0f, 2.0f });

	vext::core::cuda::allocator::deallocate(values);
	vext::core::cuda::allocator::free();
}

TEST(CoreCudaReduction, ComputesFullBufferSum)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	float* src = copy_to_device(std::vector<float>{ 1.0f, 2.0f, 3.0f, 4.0f });
	float* out = vext::core::cuda::allocator::allocate<float>(1);

	vext::core::cuda::operations::reduce<vext::core::ReductionOperation::SUM>(
		out,
		src,
		1,
		4,
		std::vector<std::uint32_t>{ 1 },
		std::vector<std::uint32_t>{ 0 },
		std::vector<std::uint32_t>{ 4 },
		std::vector<std::uint32_t>{ 1 });
	check_cuda(cudaDeviceSynchronize());

	expect_near(copy_to_host(out, 1), { 10.0f });

	vext::core::cuda::allocator::deallocate(src);
	vext::core::cuda::allocator::deallocate(out);
	vext::core::cuda::allocator::free();
}

TEST(CoreCudaLinearAlgebra, ComputesMatrixProduct)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	float* lhs = copy_to_device(std::vector<float>{ 1, 2, 3, 4, 5, 6 });
	float* rhs = copy_to_device(std::vector<float>{ 7, 8, 9, 10, 11, 12 });
	float* out = vext::core::cuda::allocator::allocate<float>(4);

	vext::core::cuda::operations::matmul(out, lhs, rhs, 2, 3, 2);
	check_cuda(cudaDeviceSynchronize());

	expect_near(copy_to_host(out, 4), { 58, 64, 139, 154 });

	vext::core::cuda::allocator::deallocate(lhs);
	vext::core::cuda::allocator::deallocate(rhs);
	vext::core::cuda::allocator::deallocate(out);
	vext::core::cuda::allocator::free();
}
