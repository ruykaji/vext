#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <vext/nn/linear.hpp>
#include <vext/nn/module.hpp>
#include <vext/tensor.hpp>

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

class ParameterModule : public vext::nn::Module<vext::Backend::CPU>
{
	VEXT_MODULE(vext::Backend::CPU);

public:
	ParameterModule()
		: first({ 1.0f, 2.0f }),
		  second({ { 3.0f, 4.0f }, { 5.0f, 6.0f } })
	{
		add<vext::Initialization::XAVIER_NORMAL>(&first);
		add<vext::Initialization::XAVIER_NORMAL>(&second);
	}

	vext::Tensor<float> first;
	vext::Tensor<float> second;
};

class ParentModule : public vext::nn::Module<vext::Backend::CPU>
{
	VEXT_MODULE(vext::Backend::CPU);

public:
	ParentModule()
		: parent_parameter({ 7.0f })
	{
		add<vext::Initialization::XAVIER_NORMAL>(&parent_parameter);

		add(&left);
		add(&right);
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

TEST(NnCpu, LinearRegistersWeightAndBiasParameters)
{
	const vext::nn::Linear<vext::Backend::CPU> linear(3, 2);

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
	const vext::nn::Linear<vext::Backend::CPU> linear(3, 2);
	const vext::Tensor<float>                  input({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

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
	const vext::nn::Linear<vext::Backend::CPU> linear(3, 2);
	const vext::Tensor<float>                  input({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });

	EXPECT_THROW((void)linear(input), std::runtime_error);
}
