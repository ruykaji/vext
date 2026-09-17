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

template <vext::EvaluationMode Mp = vext::EvaluationMode::PLAIN>
class ParameterModule : public vext::nn::Module<vext::Backend::CPU, Mp>
{
public:
	ParameterModule()
		: vext::nn::Module<vext::Backend::CPU, Mp>(first, second),
		  first(2),
		  second(2, 2)
	{
		first.tensor().set_from({ 1.0f, 2.0f });
		second.tensor().set_from({ 3.0f, 4.0f, 5.0f, 6.0f });
	}

	vext::optim::Parameter<float, vext::Backend::CPU, Mp> first;
	vext::optim::Parameter<float, vext::Backend::CPU, Mp> second;
};

class ParentModule : public vext::nn::Module<vext::Backend::CPU>
{
public:
	ParentModule()
		: vext::nn::Module<vext::Backend::CPU>(parent_parameter, left, right),
		  parent_parameter(2)
	{
		parent_parameter.tensor().set_from({ 7.0f, 8.0f });
	}

	vext::optim::Parameter<float, vext::Backend::CPU> parent_parameter;
	ParameterModule<>                                 left;
	ParameterModule<>                                 right;
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
	ParameterModule<> module;

	vext::nn::Module<vext::Backend::CPU>::iterator it = module.begin();

	ASSERT_NE(it, module.end());
	expect_shape_eq(it->tensor().dims(), { 2 });
	expect_tensor_near(it->tensor(), { 1.0f, 2.0f });

	++it;

	ASSERT_NE(it, module.end());
	expect_shape_eq(it->tensor().dims(), { 2, 2 });
	expect_tensor_near(it->tensor(), { 3.0f, 4.0f, 5.0f, 6.0f });

	++it;

	EXPECT_EQ(it, module.end());
}

TEST(NnCpu, ModuleIteratorAllowsParameterMutation)
{
	ParameterModule<>    module;
	vext::Tensor<float>& parameter = module.begin()->tensor();
	vext::binary<vext::Op::ADD>(parameter, vext::Tensor<float>({ 10.0f, 20.0f }), parameter);

	expect_tensor_near(module.first.tensor(), { 11.0f, 22.0f });
}

TEST(NnCpu, ConstModuleIteratesConstParameters)
{
	const ParameterModule<>                              module;
	vext::nn::Module<vext::Backend::CPU>::const_iterator it = module.begin();

	static_assert(std::is_const_v<std::remove_reference_t<decltype(*it)>>);

	ASSERT_NE(it, module.end());
	expect_shape_eq(it->tensor().dims(), { 2 });

	++it;

	ASSERT_NE(it, module.end());
	expect_shape_eq(it->tensor().dims(), { 2, 2 });
}

TEST(NnCpu, ModuleRecursivelyIteratesChildParameters)
{
	ParentModule module;

	std::vector<std::vector<std::uint32_t>> shapes;

	for(const auto& parameter : module)
		{
			shapes.emplace_back(parameter.tensor().dims());
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
	ParameterModule<>                              module;
	vext::nn::Module<vext::Backend::CPU>::iterator it = module.begin();

	const vext::nn::Module<vext::Backend::CPU>::iterator previous = it++;

	expect_shape_eq(previous->tensor().dims(), { 2 });
	expect_shape_eq(it->tensor().dims(), { 2, 2 });
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
	vext::nn::activation::ReLU<vext::Backend::CPU, vext::Mutation::OUT_OF_PLACE> relu;
	const vext::Tensor<float>                                                    input({ -2.0f, 0.0f, 3.0f });

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

TEST(NnCpu, XavierUniformInitializesFiniteValuesWithinExpectedBounds)
{
	vext::optim::Parameter<float, vext::Backend::CPU> parameter(4, 8);

	parameter.xavier_uniform();
	const vext::Tensor<float>& weight = parameter.tensor();

	const float sigma = 2.0f / (8.0f + 4.0f);
	const float bound = std::sqrt(3.0f * sigma);

	expect_shape_eq(weight.dims(), { 4, 8 });
	expect_tensor_finite(weight);
	expect_tensor_between(weight, -bound, bound);
}

TEST(NnCpu, XavierNormalInitializesFiniteValues)
{
	vext::optim::Parameter<float, vext::Backend::CPU> parameter(4, 8);

	parameter.xavier_normal();
	const vext::Tensor<float>& weight = parameter.tensor();

	expect_shape_eq(weight.dims(), { 4, 8 });
	expect_tensor_finite(weight);
}

TEST(NnCpu, KaimingUniformInitializesFiniteValuesWithinExpectedBounds)
{
	vext::optim::Parameter<float, vext::Backend::CPU> parameter(4, 8);
	const float                                       alpha = 0.25f;

	parameter.kaiming_uniform(alpha);
	const vext::Tensor<float>& weight = parameter.tensor();

	const float gain  = std::sqrt(2.0f / (1.0f + alpha));
	const float bound = gain * std::sqrt(3.0f / 8.0f);

	expect_shape_eq(weight.dims(), { 4, 8 });
	expect_tensor_finite(weight);
	expect_tensor_between(weight, -bound, bound);
}

TEST(NnCpu, KaimingNormalInitializesFiniteValues)
{
	vext::optim::Parameter<float, vext::Backend::CPU> parameter(4, 8);

	parameter.kaiming_normal(0.25f);
	const vext::Tensor<float>& weight = parameter.tensor();

	expect_shape_eq(weight.dims(), { 4, 8 });
	expect_tensor_finite(weight);
}

TEST(NnCpu, LinearRegistersWeightAndBiasParameters)
{
	vext::nn::layer::Linear<vext::Backend::CPU> layer(3, 4);

	vext::nn::Module<vext::Backend::CPU>::iterator it = layer.begin();

	ASSERT_NE(it, layer.end());
	expect_shape_eq(it->tensor().dims(), { 3, 4 });
	expect_tensor_finite(it->tensor());

	++it;

	ASSERT_NE(it, layer.end());
	expect_shape_eq(it->tensor().dims(), { 4 });
	expect_tensor_finite(it->tensor());

	++it;

	EXPECT_EQ(it, layer.end());
}

TEST(NnCpu, LinearForwardProducesExpectedOutputShapeAndFiniteValues)
{
	vext::nn::layer::Linear<vext::Backend::CPU> layer(3, 4);
	const vext::Tensor<float>                   input({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

	const vext::Tensor<float> output = layer(input);

	expect_shape_eq(output.dims(), { 2, 4 });
	expect_tensor_finite(output);
}

TEST(NnCpuNoise, PerturbedLinearUsesEachParametersConfiguredSeed)
{
	constexpr std::uint32_t weight_seed = 101;
	constexpr std::uint32_t bias_seed   = 202;

	vext::nn::layer::Linear<vext::Backend::CPU, vext::EvaluationMode::PLAIN> plain(2, 2);
	auto                                                                     plain_parameter = plain.begin();
	plain_parameter->tensor().set_from({ 1.0f, 1.0f, 1.0f, 1.0f });
	++plain_parameter;
	plain_parameter->tensor().set_from({ 0.0f, 0.0f });

	vext::nn::layer::Linear<vext::Backend::CPU, vext::EvaluationMode::PERTURBED> perturbed(2, 2);
	auto                                                                         perturbed_parameter = perturbed.begin();
	perturbed_parameter->tensor().set_from({ 1.0f, 1.0f, 1.0f, 1.0f });
	perturbed_parameter->set_seed({ weight_seed });
	++perturbed_parameter;
	perturbed_parameter->tensor().set_from({ 0.0f, 0.0f });
	perturbed_parameter->set_seed({ bias_seed });

	vext::core::cpu::NoiseDescriptor& descriptor = vext::core::cpu::sequentional_noise_descriptor();
	descriptor.seed                              = 0;
	descriptor.counter                           = 0;
	descriptor.direction                         = 1;

	const vext::Tensor<float> input({ { 1.0f, 2.0f } });
	expect_tensor_near(plain(input), { 3.0f, 3.0f });

	const vext::Tensor<float> output = perturbed(input);
	expect_tensor_near(output, { 1.0f * (1.0f + noise_value(weight_seed, 1, 0)) + 2.0f * (1.0f + noise_value(weight_seed, 1, 2)) + noise_value(bias_seed, 2, 0), 1.0f * (1.0f + noise_value(weight_seed, 1, 1)) + 2.0f * (1.0f + noise_value(weight_seed, 1, 3)) + noise_value(bias_seed, 2, 1) });
}
