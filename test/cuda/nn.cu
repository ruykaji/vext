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

template <vext::ParameterMode Mp = vext::ParameterMode::PLAIN>
class ParameterModule : public vext::nn::Module<vext::Backend::CUDA, Mp>
{
public:
	ParameterModule()
		: vext::nn::Module<vext::Backend::CUDA, Mp>(first, second),
		  first(2),
		  second(2, 2)
	{
		static_cast<vext::Tensor<float, vext::Backend::CUDA>&>(first).set_from({ 1.0f, 2.0f });
		static_cast<vext::Tensor<float, vext::Backend::CUDA>&>(second).set_from({ 3.0f, 4.0f, 5.0f, 6.0f });
	}

	vext::optim::Parameter<float, vext::Backend::CUDA, Mp> first;
	vext::optim::Parameter<float, vext::Backend::CUDA, Mp> second;
};

void
expect_cuda_tensor_near(
	const vext::Tensor<float, vext::Backend::CUDA>& tensor,
	const std::vector<float>&                       expected,
	const float                                     tolerance = 1e-5f)
{
	ASSERT_EQ(tensor.length(), expected.size());

	for(std::uint32_t i = 0; i < tensor.length(); ++i)
		{
			EXPECT_NEAR(tensor.item(i), expected[i], tolerance) << "at flat index " << i;
		}
}

void
expect_cuda_tensor_finite(
	const vext::Tensor<float, vext::Backend::CUDA>& tensor)
{
	for(std::uint32_t i = 0; i < tensor.length(); ++i)
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
	for(std::uint32_t i = 0; i < tensor.length(); ++i)
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

	ParameterModule<>                               module;
	vext::nn::Module<vext::Backend::CUDA>::iterator it = module.begin();

	ASSERT_NE(it, module.end());
	EXPECT_EQ((static_cast<const vext::Tensor<float, vext::Backend::CUDA>&>(*it).dims()), (std::vector<std::uint32_t>{ 2 }));

	++it;

	ASSERT_NE(it, module.end());
	EXPECT_EQ((static_cast<const vext::Tensor<float, vext::Backend::CUDA>&>(*it).dims()), (std::vector<std::uint32_t>{ 2, 2 }));

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

	vext::optim::Parameter<float, vext::Backend::CUDA> uniform_parameter(4, 8);
	vext::optim::Parameter<float, vext::Backend::CUDA> normal_parameter(4, 8);

	uniform_parameter.xavier_uniform();
	normal_parameter.kaiming_normal(0.25f);
	const vext::Tensor<float, vext::Backend::CUDA>& uniform_weight = uniform_parameter;
	const vext::Tensor<float, vext::Backend::CUDA>& normal_weight  = normal_parameter;

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
	EXPECT_EQ((static_cast<const vext::Tensor<float, vext::Backend::CUDA>&>(*it).dims()), (std::vector<std::uint32_t>{ 3, 4 }));
	expect_cuda_tensor_finite(*it);

	++it;

	ASSERT_NE(it, layer.end());
	EXPECT_EQ((static_cast<const vext::Tensor<float, vext::Backend::CUDA>&>(*it).dims()), (std::vector<std::uint32_t>{ 4 }));
	expect_cuda_tensor_finite(*it);

	const vext::Tensor<float, vext::Backend::CUDA> input({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
	const vext::Tensor<float, vext::Backend::CUDA> output = layer(input);

	EXPECT_EQ(output.dims(), (std::vector<std::uint32_t>{ 2, 4 }));
	expect_cuda_tensor_finite(output);
}

TEST(NnCudaNoise, PerturbedLinearUsesEachParametersConfiguredSeed)
{
	if(!has_cuda_device())
		{
			GTEST_SKIP() << "No CUDA-capable device is available";
		}

	constexpr std::uint32_t weight_seed = 101;
	constexpr std::uint32_t bias_seed   = 202;

	vext::nn::layer::Linear<vext::Backend::CUDA, vext::ParameterMode::PERTURBED> layer(2, 2);
	auto                                                                         parameter = layer.begin();
	static_cast<vext::Tensor<float, vext::Backend::CUDA>&>(*parameter).set_from({ 1.0f, 1.0f, 1.0f, 1.0f });
	parameter->set_seed({ weight_seed });
	++parameter;
	static_cast<vext::Tensor<float, vext::Backend::CUDA>&>(*parameter).set_from({ 0.0f, 0.0f });
	parameter->set_seed({ bias_seed });

	vext::core::cuda::NoiseDescriptor& descriptor = vext::core::cuda::sequentional_noise_descriptor();
	descriptor.seed                               = 0;
	descriptor.counter                            = 0;
	descriptor.direction                          = 1;

	const vext::Tensor<float, vext::Backend::CUDA> input({ { 1.0f, 2.0f } });
	const vext::Tensor<float, vext::Backend::CUDA> output = layer(input);
	expect_cuda_tensor_near(output, { 1.0f * (1.0f + noise_value(weight_seed, 1, 0)) + 2.0f * (1.0f + noise_value(weight_seed, 1, 2)) + noise_value(bias_seed, 2, 0), 1.0f * (1.0f + noise_value(weight_seed, 1, 1)) + 2.0f * (1.0f + noise_value(weight_seed, 1, 3)) + noise_value(bias_seed, 2, 1) });
}
