#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include <vext/core/cuda/allocator.cuh>
#include <vext/core/cuda/operations/csr_scatter.cuh>
#include <vext/core/cuda/operations/csr_spmv.cuh>
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

TEST(CoreCudaCsrScatter, ComputesSegmentAggregates)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const std::vector<float> src_values{
		1.0f, 2.0f,
		3.0f, 4.0f,
		-2.0f, 5.0f,
		4.0f, -1.0f
	};
	const std::vector<std::uint32_t> head_values{ 0, 2, 4, 4, 4 };
	const std::vector<std::uint32_t> tail_values{ 0, 2, 1, 3 };

	float*         src  = copy_to_device(src_values);
	std::uint32_t* head = copy_to_device(head_values);
	std::uint32_t* tail = copy_to_device(tail_values);
	float*         out  = vext::core::cuda::allocator::allocate<float>(8);

	check_cuda(cudaMemset(out, 0, 8 * sizeof(float)));
	vext::core::cuda::operations::csr_scatter<vext::core::CSRScatterOperation::SUM>(out, src, head, tail, 4, 2);
	check_cuda(cudaDeviceSynchronize());
	expect_near(copy_to_host(out, 8), { -1.0f, 7.0f, 7.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f });

	check_cuda(cudaMemset(out, 0, 8 * sizeof(float)));
	vext::core::cuda::operations::csr_scatter<vext::core::CSRScatterOperation::MEAN>(out, src, head, tail, 4, 2);
	check_cuda(cudaDeviceSynchronize());
	expect_near(copy_to_host(out, 8), { -0.5f, 3.5f, 3.5f, 1.5f, 0.0f, 0.0f, 0.0f, 0.0f });

	check_cuda(cudaMemset(out, 0, 8 * sizeof(float)));
	vext::core::cuda::operations::csr_scatter<vext::core::CSRScatterOperation::MAX>(out, src, head, tail, 4, 2);
	check_cuda(cudaDeviceSynchronize());
	expect_near(copy_to_host(out, 8), { 1.0f, 5.0f, 4.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f });

	check_cuda(cudaMemset(out, 0, 8 * sizeof(float)));
	vext::core::cuda::operations::csr_scatter<vext::core::CSRScatterOperation::VAR>(out, src, head, tail, 4, 2);
	check_cuda(cudaDeviceSynchronize());
	expect_near(copy_to_host(out, 8), { 2.25f, 2.25f, 0.25f, 6.25f, 0.0f, 0.0f, 0.0f, 0.0f });

	check_cuda(cudaMemset(out, 0, 8 * sizeof(float)));
	vext::core::cuda::operations::csr_scatter<vext::core::CSRScatterOperation::STD>(out, src, head, tail, 4, 2);
	check_cuda(cudaDeviceSynchronize());
	expect_near(copy_to_host(out, 8), { 1.5f, 1.5f, 0.5f, 2.5f, 0.0f, 0.0f, 0.0f, 0.0f });

	vext::core::cuda::allocator::deallocate(src);
	vext::core::cuda::allocator::deallocate(head);
	vext::core::cuda::allocator::deallocate(tail);
	vext::core::cuda::allocator::deallocate(out);
	vext::core::cuda::allocator::free();
}

TEST(CoreCudaCsrScatter, ComputesMinAndProductWhenOutputIsInitializedForTheOperation)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const std::vector<float> src_values{
		1.0f, 2.0f,
		3.0f, 4.0f,
		-2.0f, 5.0f,
		4.0f, -1.0f
	};
	const std::vector<std::uint32_t> head_values{ 0, 2, 4, 4, 4 };
	const std::vector<std::uint32_t> tail_values{ 0, 2, 1, 3 };

	float*         src     = copy_to_device(src_values);
	std::uint32_t* head    = copy_to_device(head_values);
	std::uint32_t* tail    = copy_to_device(tail_values);
	float*         min_out = copy_to_device(std::vector<float>(8, std::numeric_limits<float>::max()));
	float*         prod_out = copy_to_device(std::vector<float>(8, 1.0f));

	vext::core::cuda::operations::csr_scatter<vext::core::CSRScatterOperation::MIN>(min_out, src, head, tail, 4, 2);
	check_cuda(cudaDeviceSynchronize());
	expect_near(copy_to_host(min_out, 8), { -2.0f, 2.0f, 3.0f, -1.0f, std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() });

	vext::core::cuda::operations::csr_scatter<vext::core::CSRScatterOperation::PROD>(prod_out, src, head, tail, 4, 2);
	check_cuda(cudaDeviceSynchronize());
	expect_near(copy_to_host(prod_out, 8), { -2.0f, 10.0f, 12.0f, -4.0f, 1.0f, 1.0f, 1.0f, 1.0f });

	vext::core::cuda::allocator::deallocate(src);
	vext::core::cuda::allocator::deallocate(head);
	vext::core::cuda::allocator::deallocate(tail);
	vext::core::cuda::allocator::deallocate(min_out);
	vext::core::cuda::allocator::deallocate(prod_out);
	vext::core::cuda::allocator::free();
}

TEST(CoreCudaCsrSpmv, ComputesSparseMatrixVectorAggregates)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const std::vector<float>         values_host{ 1.0f, 2.0f, 3.0f, 4.0f, -2.0f, 5.0f };
	const std::vector<std::uint32_t> head_host{ 0, 2, 4, 6 };
	const std::vector<std::uint32_t> tail_host{ 0, 2, 1, 3, 0, 1 };
	const std::vector<float>         x_host{ 2.0f, -1.0f, 3.0f, 4.0f };

	float*         values = copy_to_device(values_host);
	std::uint32_t* head   = copy_to_device(head_host);
	std::uint32_t* tail   = copy_to_device(tail_host);
	float*         x      = copy_to_device(x_host);
	float*         out    = vext::core::cuda::allocator::allocate<float>(3);

	vext::core::cuda::operations::csr_spmv<vext::core::CSRSpMVOperation::SUM>(out, values, head, tail, x, 3);
	check_cuda(cudaDeviceSynchronize());
	expect_near(copy_to_host(out, 3), { 8.0f, 13.0f, -9.0f });

	vext::core::cuda::operations::csr_spmv<vext::core::CSRSpMVOperation::MEAN>(out, values, head, tail, x, 3);
	check_cuda(cudaDeviceSynchronize());
	expect_near(copy_to_host(out, 3), { 4.0f, 6.5f, -4.5f });

	vext::core::cuda::operations::csr_spmv<vext::core::CSRSpMVOperation::MIN>(out, values, head, tail, x, 3);
	check_cuda(cudaDeviceSynchronize());
	expect_near(copy_to_host(out, 3), { 2.0f, -3.0f, -5.0f });

	vext::core::cuda::operations::csr_spmv<vext::core::CSRSpMVOperation::MAX>(out, values, head, tail, x, 3);
	check_cuda(cudaDeviceSynchronize());
	expect_near(copy_to_host(out, 3), { 6.0f, 16.0f, -4.0f });

	vext::core::cuda::operations::csr_spmv<vext::core::CSRSpMVOperation::PROD>(out, values, head, tail, x, 3);
	check_cuda(cudaDeviceSynchronize());
	expect_near(copy_to_host(out, 3), { 12.0f, -48.0f, 20.0f });

	vext::core::cuda::operations::csr_spmv<vext::core::CSRSpMVOperation::VAR>(out, values, head, tail, x, 3);
	check_cuda(cudaDeviceSynchronize());
	expect_near(copy_to_host(out, 3), { 4.0f, 90.25f, 0.25f });

	vext::core::cuda::operations::csr_spmv<vext::core::CSRSpMVOperation::STD>(out, values, head, tail, x, 3);
	check_cuda(cudaDeviceSynchronize());
	expect_near(copy_to_host(out, 3), { 2.0f, 9.5f, 0.5f });

	vext::core::cuda::allocator::deallocate(values);
	vext::core::cuda::allocator::deallocate(head);
	vext::core::cuda::allocator::deallocate(tail);
	vext::core::cuda::allocator::deallocate(x);
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
