#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
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
	const vext::Shape&                         shape,
	const std::initializer_list<std::uint64_t> expected)
{
	ASSERT_EQ(shape.dims(), std::vector<std::uint64_t>(expected));
	ASSERT_EQ(shape.size(), expected.size());
}

template <typename T1, typename T2>
void
expect_tensor_eq(
	const vext::Tensor<T1>& actual,
	const vext::Tensor<T2>& expected)
{
	ASSERT_EQ(actual.shape(), expected.shape());
	ASSERT_EQ((actual == expected).min().item(0), 1);
}

template <typename T1, typename T2>
void
expect_tensor_near(
	const vext::Tensor<T1>& actual,
	const vext::Tensor<T2>& expected,
	const float             tolerance = 1e-5f)
{
	ASSERT_EQ(actual.shape(), expected.shape());

	for(std::uint64_t i = 0; i < actual.shape().length(); ++i)
		{
			EXPECT_NEAR(static_cast<float>(actual.flat_item(i)), static_cast<float>(expected.flat_item(i)), tolerance);
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
		: parent_parameter(1)
	{
		parent_parameter.item(0) = 7.0f;

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

	auto it = module.begin();

	ASSERT_NE(it, module.end());
	expect_tensor_eq(*it, module.first);

	++it;

	ASSERT_NE(it, module.end());
	expect_tensor_eq(*it, module.second);

	++it;

	EXPECT_EQ(it, module.end());
}

TEST(NnCpu, ModuleIteratorAllowsParameterMutation)
{
	ParameterModule module;
	module.begin()->item(1) = 42.0f;

	expect_tensor_eq(module.first, vext::Tensor<float>({ 1.0f, 42.0f }));
}

TEST(NnCpu, ConstModuleIteratesConstParameters)
{
	const ParameterModule module;
	auto                  it = module.begin();

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

	std::vector<vext::Shape> shapes;

	for(const auto& parameter : module)
		{
			shapes.emplace_back(parameter.shape());
		}

	ASSERT_EQ(shapes.size(), 5);
	expect_shape_eq(shapes[0], { 1 });
	expect_shape_eq(shapes[1], { 2 });
	expect_shape_eq(shapes[2], { 2, 2 });
	expect_shape_eq(shapes[3], { 2 });
	expect_shape_eq(shapes[4], { 2, 2 });
}

TEST(NnCpu, ModuleIteratorPostIncrementReturnsPreviousParameter)
{
	ParameterModule module;
	auto            it = module.begin();

	const auto previous = it++;

	expect_tensor_eq(*previous, module.first);
	expect_tensor_eq(*it, module.second);
}

TEST(NnCpu, ModuleIteratorDefaultConstructedValuesCompareEqual)
{
	const vext::nn::Module<vext::Backend::CPU>::iterator lhs;
	const vext::nn::Module<vext::Backend::CPU>::iterator rhs;

	EXPECT_EQ(lhs, rhs);
}

TEST(NnCpu, LinearRegistersWeightAndBiasParameters)
{
	const vext::nn::layer::Linear<vext::Backend::CPU> linear(3, 2);

	auto it = linear.begin();

	ASSERT_NE(it, linear.end());
	expect_shape_eq(it->shape(), { 3, 2 });

	++it;

	ASSERT_NE(it, linear.end());
	expect_shape_eq(it->shape(), { 2 });

	++it;

	EXPECT_EQ(it, linear.end());
}

TEST(NnCpu, LinearComputesInputMatmulWeightPlusBias)
{
	const vext::nn::layer::Linear<vext::Backend::CPU> linear(3, 2);
	const vext::Tensor<float>                         input({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

	auto it = linear.begin();

	ASSERT_NE(it, linear.end());

	const vext::Tensor<float> weight(*it);

	++it;

	ASSERT_NE(it, linear.end());

	const vext::Tensor<float> bias(*it);

	const auto output   = linear(input);
	const auto expected = input.matmul(weight) + bias;

	expect_shape_eq(output.shape(), { 2, 2 });
	expect_tensor_eq(output, expected);
}

TEST(NnCpu, LinearRejectsInputsWithIncompatibleLastDimension)
{
	const vext::nn::layer::Linear<vext::Backend::CPU> linear(3, 2);
	const vext::Tensor<float>                         input({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });

	EXPECT_THROW((void)linear(input), std::runtime_error);
}

TEST(NnCpu, ActivationReluMutatesInputInPlace)
{
	vext::nn::activation::ReLU<vext::Backend::CPU> relu;
	vext::Tensor<float>                            input({ -2.0f, 0.0f, 3.0f });

	relu(input);

	expect_tensor_near(input, vext::Tensor<float>({ 0.0f, 0.0f, 3.0f }));
}

TEST(NnCpu, ActivationReluCopyReturnsTransformedCopyAndLeavesInputUnchanged)
{
	vext::nn::activation::ReLU<vext::Backend::CPU, vext::Mutation::COPY> relu;
	const vext::Tensor<float>                                            input({ -2.0f, 0.0f, 3.0f });

	const auto output = relu(input);

	expect_tensor_near(output, vext::Tensor<float>({ 0.0f, 0.0f, 3.0f }));
	expect_tensor_eq(input, vext::Tensor<float>({ -2.0f, 0.0f, 3.0f }));
}

TEST(NnCpu, ActivationSigmoidMutatesInputInPlace)
{
	vext::nn::activation::Sigmoid<vext::Backend::CPU> sigmoid;
	vext::Tensor<float>                               input({ 0.0f, 2.0f });

	sigmoid(input);

	expect_tensor_near(input, vext::Tensor<float>({ 0.5f, 1.0f / (1.0f + std::exp(-2.0f)) }));
}

TEST(NnCpu, ActivationSoftmaxMutatesInputInPlace)
{
	vext::nn::activation::Softmax<vext::Backend::CPU> softmax;
	vext::Tensor<float>                               input({ 1.0f, 2.0f, 3.0f });
	const float                                       sum = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);

	softmax(input);

	expect_tensor_near(input, vext::Tensor<float>({ std::exp(1.0f) / sum, std::exp(2.0f) / sum, std::exp(3.0f) / sum }));
}

TEST(NnCpu, ActivationLeakyReluUsesProvidedSlope)
{
	vext::nn::activation::LeakyReLU<vext::Backend::CPU> leaky_relu;
	vext::Tensor<float>                                 input({ -2.0f, 3.0f });

	leaky_relu(input, 0.25f);

	expect_tensor_near(input, vext::Tensor<float>({ -0.5f, 3.0f }));
}

TEST(NnCpu, ActivationEluUsesProvidedAlpha)
{
	vext::nn::activation::ELU<vext::Backend::CPU> elu;
	vext::Tensor<float>                           input({ -1.0f, 2.0f });

	elu(input, 2.0f);

	expect_tensor_near(input, vext::Tensor<float>({ 2.0f * (std::exp(-1.0f) - 1.0f), 2.0f }));
}

TEST(NnCpu, CalculateFanInAndFanOutHandlesVectorsMatricesAndKernels)
{
	EXPECT_EQ(vext::nn::utils::calculate_fan_in_and_fan_out(vext::Shape(5)), (std::pair<std::uint64_t, std::uint64_t>{ 1, 5 }));
	EXPECT_EQ(vext::nn::utils::calculate_fan_in_and_fan_out(vext::Shape(3, 2)), (std::pair<std::uint64_t, std::uint64_t>{ 2, 3 }));
	EXPECT_EQ(vext::nn::utils::calculate_fan_in_and_fan_out(vext::Shape(16, 3, 5, 5)), (std::pair<std::uint64_t, std::uint64_t>{ 75, 400 }));
}

TEST(NnCpu, XavierUniformInitializesEveryValueWithinExpectedBounds)
{
	vext::Tensor<float> weight(3, 2);
	const auto [fan_in, fan_out] = vext::nn::utils::calculate_fan_in_and_fan_out(weight.shape());
	const float bound            = std::sqrt(3.0f * (2.0f / (fan_in + fan_out)));

	vext::nn::utils::xavier_uniform(weight);

	for(std::uint64_t i = 0; i < weight.shape().length(); ++i)
		{
			EXPECT_GE(weight.flat_item(i), -bound);
			EXPECT_LE(weight.flat_item(i), bound);
		}
}

TEST(NnCpu, KaimingUniformInitializesEveryValueWithinExpectedBounds)
{
	vext::Tensor<float> weight(3, 2);
	const auto [fan_in, _] = vext::nn::utils::calculate_fan_in_and_fan_out(weight.shape());
	const float gain       = std::sqrt(2.0f);
	const float bound      = gain * std::sqrt(3.0f / fan_in);

	vext::nn::utils::kaiming_uniform(weight);

	for(std::uint64_t i = 0; i < weight.shape().length(); ++i)
		{
			EXPECT_GE(weight.flat_item(i), -bound);
			EXPECT_LE(weight.flat_item(i), bound);
		}
}

TEST(NnCpu, NormalInitializersPreserveTensorShape)
{
	vext::Tensor<float> xavier(3, 2);
	vext::Tensor<float> kaiming(4, 3);

	vext::nn::utils::xavier_normal(xavier);
	vext::nn::utils::kaiming_normal(kaiming, 0.25f);

	expect_shape_eq(xavier.shape(), { 3, 2 });
	expect_shape_eq(kaiming.shape(), { 4, 3 });
}
