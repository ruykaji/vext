#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <vector>

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

void
expect_shape_eq(
	const std::vector<std::uint32_t>&          shape,
	const std::initializer_list<std::uint32_t> expected)
{
	ASSERT_EQ(shape, std::vector<std::uint32_t>(expected));
	ASSERT_EQ(shape.size(), expected.size());
}

void
expect_tensor_near(
	const vext::Tensor<float>&         tensor,
	const std::initializer_list<float> expected,
	const float                        tolerance = 1e-5f)
{
	ASSERT_EQ(tensor.length(), expected.size());

	std::uint32_t i = 0;
	for(const auto value : expected)
		{
			EXPECT_NEAR(tensor.item(i), value, tolerance) << "at flat index " << i;
			++i;
		}
}

void
expect_tensor_finite(
	const vext::Tensor<float>& tensor)
{
	for(std::uint32_t i = 0; i < tensor.length(); ++i)
		{
			EXPECT_TRUE(std::isfinite(tensor.item(i))) << "at flat index " << i;
		}
}

void
expect_tensor_between(
	const vext::Tensor<float>& tensor,
	const float                lower,
	const float                upper)
{
	for(std::uint32_t i = 0; i < tensor.length(); ++i)
		{
			EXPECT_GE(tensor.item(i), lower) << "at flat index " << i;
			EXPECT_LE(tensor.item(i), upper) << "at flat index " << i;
		}
}

class ParameterModule : public vext::nn::Module<vext::Backend::CPU>
{
	VEXT_MODULE(vext::Backend::CPU);

public:
	ParameterModule()
		: first({ 1.0f, 2.0f }),
		  second({ { 3.0f, 4.0f }, { 5.0f, 6.0f } })
	{
		assign_parameter(&first);
		assign_parameter(&second);
	}

	vext::Tensor<float> first;
	vext::Tensor<float> second;
};

class ParentModule : public vext::nn::Module<vext::Backend::CPU>
{
	VEXT_MODULE(vext::Backend::CPU);

public:
	ParentModule()
		: parent_parameter({ 7.0f, 8.0f })
	{
		assign_parameter(&parent_parameter);
		assign_module(&left);
		assign_module(&right);
	}

	vext::Tensor<float> parent_parameter;
	ParameterModule     left;
	ParameterModule     right;
};

}

TEST(NnCpu, EmptyModuleHasNoParameters)
{
	const vext::nn::Module<vext::Backend::CPU> module;

	EXPECT_EQ(module.begin(), module.end());
	EXPECT_EQ(std::distance(module.begin(), module.end()), 0);
}

TEST(NnCpu, ModuleIteratesRegisteredParameters)
{
	ParameterModule module;

	vext::nn::Module<vext::Backend::CPU>::iterator it = module.begin();

	ASSERT_NE(it, module.end());
	expect_shape_eq(it->shape(), { 2 });
	expect_tensor_near(*it, { 1.0f, 2.0f });

	++it;

	ASSERT_NE(it, module.end());
	expect_shape_eq(it->shape(), { 2, 2 });
	expect_tensor_near(*it, { 3.0f, 4.0f, 5.0f, 6.0f });

	++it;

	EXPECT_EQ(it, module.end());
}

TEST(NnCpu, ModuleIteratorAllowsParameterMutation)
{
	ParameterModule module;
	*module.begin() += vext::Tensor<float>({ 10.0f, 20.0f });

	expect_tensor_near(module.first, { 11.0f, 22.0f });
}

TEST(NnCpu, ConstModuleIteratesConstParameters)
{
	const ParameterModule                                module;
	vext::nn::Module<vext::Backend::CPU>::const_iterator it = module.begin();

	static_assert(std::is_const_v<std::remove_reference_t<decltype(*it)>>);

	ASSERT_NE(it, module.end());
	expect_shape_eq(it->shape(), { 2 });

	++it;

	ASSERT_NE(it, module.end());
	expect_shape_eq(it->shape(), { 2, 2 });
}

TEST(NnCpu, ModuleRecursivelyIteratesChildParameters)
{
	ParentModule module;

	std::vector<std::vector<std::uint32_t>> shapes;

	for(const auto& parameter : module)
		{
			shapes.emplace_back(parameter.shape());
		}

	ASSERT_EQ(shapes.size(), 5);
	expect_shape_eq(shapes[0], { 2 });
	expect_shape_eq(shapes[1], { 2 });
	expect_shape_eq(shapes[2], { 2, 2 });
	expect_shape_eq(shapes[3], { 2 });
	expect_shape_eq(shapes[4], { 2, 2 });
}

TEST(NnCpu, ModuleIteratorPostIncrementReturnsPreviousParameter)
{
	ParameterModule                                module;
	vext::nn::Module<vext::Backend::CPU>::iterator it = module.begin();

	const vext::nn::Module<vext::Backend::CPU>::iterator previous = it++;

	expect_shape_eq(previous->shape(), { 2 });
	expect_shape_eq(it->shape(), { 2, 2 });
}

TEST(NnCpu, ModuleIteratorDefaultConstructedValuesCompareEqual)
{
	const vext::nn::Module<vext::Backend::CPU>::iterator lhs;
	const vext::nn::Module<vext::Backend::CPU>::iterator rhs;

	EXPECT_EQ(lhs, rhs);
}

TEST(NnCpu, ActivationReluMutatesInputInPlace)
{
	vext::nn::activation::ReLU<vext::Backend::CPU> relu;
	vext::Tensor<float>                            input({ -2.0f, 0.0f, 3.0f });

	relu(input);

	expect_tensor_near(input, { 0.0f, 0.0f, 3.0f });
}

TEST(NnCpu, ActivationReluCopyReturnsTransformedCopyAndLeavesInputUnchanged)
{
	vext::nn::activation::ReLU<vext::Backend::CPU, vext::Mutation::COPY> relu;
	const vext::Tensor<float>                                            input({ -2.0f, 0.0f, 3.0f });

	const vext::Tensor<float> output = relu(input);

	expect_tensor_near(output, { 0.0f, 0.0f, 3.0f });
	expect_tensor_near(input, { -2.0f, 0.0f, 3.0f });
}

TEST(NnCpu, ActivationSigmoidMutatesInputInPlace)
{
	vext::nn::activation::Sigmoid<vext::Backend::CPU> sigmoid;
	vext::Tensor<float>                               input({ 0.0f, 2.0f });

	sigmoid(input);

	expect_tensor_near(input, { 0.5f, 1.0f / (1.0f + std::exp(-2.0f)) });
}

TEST(NnCpu, ActivationSoftmaxMutatesInputInPlace)
{
	vext::nn::activation::Softmax<vext::Backend::CPU> softmax;
	vext::Tensor<float>                               input({ 1.0f, 2.0f, 3.0f });
	const float                                       sum = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);

	softmax(input);

	expect_tensor_near(input, { std::exp(1.0f) / sum, std::exp(2.0f) / sum, std::exp(3.0f) / sum });
}

TEST(NnCpu, ActivationLeakyReluUsesProvidedSlope)
{
	vext::nn::activation::LeakyReLU<vext::Backend::CPU> leaky_relu;
	vext::Tensor<float>                                 input({ -2.0f, 3.0f });

	leaky_relu(input, 0.25f);

	expect_tensor_near(input, { -0.5f, 3.0f });
}

TEST(NnCpu, ActivationEluUsesProvidedAlpha)
{
	vext::nn::activation::ELU<vext::Backend::CPU> elu;
	vext::Tensor<float>                           input({ -1.0f, 2.0f });

	elu(input, 2.0f);

	expect_tensor_near(input, { 2.0f * (std::exp(-1.0f) - 1.0f), 2.0f });
}

TEST(NnCpu, CalculateFanInAndFanOutHandlesVectorsMatricesAndKernels)
{
	EXPECT_EQ(vext::nn::utils::calculate_fan_in_and_fan_out({ 5 }), (std::pair<std::uint64_t, std::uint64_t>{ 1, 5 }));
	EXPECT_EQ(vext::nn::utils::calculate_fan_in_and_fan_out({ 3, 2 }), (std::pair<std::uint64_t, std::uint64_t>{ 2, 3 }));
	EXPECT_EQ(vext::nn::utils::calculate_fan_in_and_fan_out({ 16, 3, 5, 5 }), (std::pair<std::uint64_t, std::uint64_t>{ 75, 400 }));
}

TEST(NnCpu, XavierUniformInitializesFiniteValuesWithinExpectedBounds)
{
	vext::Tensor<float> weight(4, 8);

	vext::nn::utils::xavier_uniform(weight);

	const float sigma = 2.0f / (8.0f + 4.0f);
	const float bound = std::sqrt(3.0f * sigma);

	expect_shape_eq(weight.shape(), { 4, 8 });
	expect_tensor_finite(weight);
	expect_tensor_between(weight, -bound, bound);
}

TEST(NnCpu, XavierNormalInitializesFiniteValues)
{
	vext::Tensor<float> weight(4, 8);

	vext::nn::utils::xavier_normal(weight);

	expect_shape_eq(weight.shape(), { 4, 8 });
	expect_tensor_finite(weight);
}

TEST(NnCpu, KaimingUniformInitializesFiniteValuesWithinExpectedBounds)
{
	vext::Tensor<float> weight(4, 8);
	const float         alpha = 0.25f;

	vext::nn::utils::kaiming_uniform(weight, alpha);

	const float gain  = std::sqrt(2.0f / (1.0f + alpha));
	const float bound = gain * std::sqrt(3.0f / 8.0f);

	expect_shape_eq(weight.shape(), { 4, 8 });
	expect_tensor_finite(weight);
	expect_tensor_between(weight, -bound, bound);
}

TEST(NnCpu, KaimingNormalInitializesFiniteValues)
{
	vext::Tensor<float> weight(4, 8);

	vext::nn::utils::kaiming_normal(weight, 0.25f);

	expect_shape_eq(weight.shape(), { 4, 8 });
	expect_tensor_finite(weight);
}

TEST(NnCpu, LinearRegistersWeightAndBiasParameters)
{
	vext::nn::layer::Linear<vext::Backend::CPU> layer(3, 4);

	vext::nn::Module<vext::Backend::CPU>::iterator it = layer.begin();

	ASSERT_NE(it, layer.end());
	expect_shape_eq(it->shape(), { 3, 4 });
	expect_tensor_finite(*it);

	++it;

	ASSERT_NE(it, layer.end());
	expect_shape_eq(it->shape(), { 4 });
	expect_tensor_finite(*it);

	++it;

	EXPECT_EQ(it, layer.end());
}

TEST(NnCpu, LinearForwardProducesExpectedOutputShapeAndFiniteValues)
{
	vext::nn::layer::Linear<vext::Backend::CPU> layer(3, 4);
	const vext::Tensor<float>                   input({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

	const vext::Tensor<float> output = layer(input);

	expect_shape_eq(output.shape(), { 2, 4 });
	expect_tensor_finite(output);
}
