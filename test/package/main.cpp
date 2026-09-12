#include <vext/ops.hpp>

int
main()
{
	const vext::Tensor<float> lhs({ 1.0f, 2.0f });
	const vext::Tensor<float> rhs({ 3.0f, 4.0f });
	const auto                result = vext::binary<vext::Op::ADD>(lhs, rhs);

	return result.item(0) == 4.0f && result.item(1) == 6.0f ? 0 : 1;
}
