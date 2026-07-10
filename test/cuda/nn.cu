#include <gtest/gtest.h>

#include <cmath>
#include <iterator>
#include <vector>

#include <cuda_runtime.h>

#include <vext/nn/activation/elu.hpp>
#include <vext/nn/activation/leaky_relu.hpp>
#include <vext/nn/activation/relu.hpp>
#include <vext/nn/activation/sigmoid.hpp>
#include <vext/nn/activation/softmax.hpp>
#include <vext/nn/layer/linear.hpp>
#include <vext/nn/module.hpp>
#include <vext/nn/utils/init.hpp>

namespace
{

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

class ParameterModule : public vext::nn::Module<vext::Backend::CUDA>
{
	VEXT_MODULE(vext::Backend::CUDA);

public:
	ParameterModule()
		: first({ 1.0f, 2.0f }),
		  second({ { 3.0f, 4.0f }, { 5.0f, 6.0f } })
	{
		assign_parameter(&first);
		assign_parameter(&second);
	}

	vext::Tensor<float, vext::Backend::CUDA> first;
	vext::Tensor<float, vext::Backend::CUDA> second;
};

void
expect_cuda_tensor_near(
	const vext::Tensor<float, vext::Backend::CUDA>& tensor,
	const std::vector<float>&                       expected,
	const float                                     tolerance = 1e-5f)
{
	ASSERT_EQ(tensor.shape().length(), expected.size());

	for(std::uint32_t i = 0; i < tensor.shape().length(); ++i)
		{
			EXPECT_NEAR(tensor.item(i), expected[i], tolerance) << "at flat index " << i;
		}
}

void
expect_cuda_tensor_finite(
	const vext::Tensor<float, vext::Backend::CUDA>& tensor)
{
	for(std::uint32_t i = 0; i < tensor.shape().length(); ++i)
		{
			EXPECT_TRUE(std::isfinite(tensor.item(i))) << "at flat index " << i;
		}
}

void
expect_cuda_tensor_between(
	const vext::Tensor<float, vext::Backend::CUDA>& tensor,
	const float                                     lower,
	const float                                     upper)
{
	for(std::uint32_t i = 0; i < tensor.shape().length(); ++i)
		{
			EXPECT_GE(tensor.item(i), lower) << "at flat index " << i;
			EXPECT_LE(tensor.item(i), upper) << "at flat index " << i;
		}
}

}

TEST(NnCuda, EmptyModuleHasNoParameters)
{
	const vext::nn::Module<vext::Backend::CUDA> module;

	EXPECT_EQ(module.begin(), module.end());
	EXPECT_EQ(std::distance(module.begin(), module.end()), 0);
}

TEST(NnCuda, ModuleIteratesRegisteredParameters)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	ParameterModule module;
	vext::nn::Module<vext::Backend::CUDA>::iterator it = module.begin();

	ASSERT_NE(it, module.end());
	EXPECT_EQ(it->shape().dims(), (std::vector<std::uint32_t>{ 2 }));

	++it;

	ASSERT_NE(it, module.end());
	EXPECT_EQ(it->shape().dims(), (std::vector<std::uint32_t>{ 2, 2 }));

	++it;
	EXPECT_EQ(it, module.end());
}

TEST(NnCuda, ActivationModulesMutateInPlace)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA> input({ -2.0f, 0.0f, 3.0f });

	vext::nn::activation::ReLU<vext::Backend::CUDA>{}(input);
	expect_cuda_tensor_near(input, { 0.0f, 0.0f, 3.0f });

	vext::nn::activation::LeakyReLU<vext::Backend::CUDA>{}(input, 0.25f);
	expect_cuda_tensor_near(input, { 0.0f, 0.0f, 3.0f });
}

TEST(NnCuda, CopyMutationActivationLeavesOriginalUnchanged)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	const vext::Tensor<float, vext::Backend::CUDA> input({ -2.0f, 3.0f });

	const vext::Tensor<float, vext::Backend::CUDA> output = vext::nn::activation::ELU<vext::Backend::CUDA, vext::Mutation::COPY>{}(input, 2.0f);

	expect_cuda_tensor_near(input, { -2.0f, 3.0f });
	expect_cuda_tensor_near(output, { 2.0f * (std::exp(-2.0f) - 1.0f), 3.0f });
}

TEST(NnCuda, InitializersWriteFiniteCudaTensorValues)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::Tensor<float, vext::Backend::CUDA> uniform_weight(4, 8);
	vext::Tensor<float, vext::Backend::CUDA> normal_weight(4, 8);

	vext::nn::utils::xavier_uniform(uniform_weight);
	vext::nn::utils::kaiming_normal(normal_weight, 0.25f);

	const float sigma = 2.0f / (8.0f + 4.0f);
	const float bound = std::sqrt(3.0f * sigma);

	expect_cuda_tensor_finite(uniform_weight);
	expect_cuda_tensor_between(uniform_weight, -bound, bound);
	expect_cuda_tensor_finite(normal_weight);
}

TEST(NnCuda, LinearRegistersParametersAndRunsForward)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	vext::nn::layer::Linear<vext::Backend::CUDA> layer(3, 4);

	vext::nn::Module<vext::Backend::CUDA>::iterator it = layer.begin();

	ASSERT_NE(it, layer.end());
	EXPECT_EQ(it->shape().dims(), (std::vector<std::uint32_t>{ 3, 4 }));
	expect_cuda_tensor_finite(*it);

	++it;

	ASSERT_NE(it, layer.end());
	EXPECT_EQ(it->shape().dims(), (std::vector<std::uint32_t>{ 4 }));
	expect_cuda_tensor_finite(*it);

	const vext::Tensor<float, vext::Backend::CUDA> input({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float, vext::Backend::CUDA> output = layer(input);

	EXPECT_EQ(output.shape().dims(), (std::vector<std::uint32_t>{ 2, 4 }));
	expect_cuda_tensor_finite(output);
}
