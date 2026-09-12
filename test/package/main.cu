#include <vext/ops.hpp>

int
main()
{
	const vext::Tensor<float, vext::Backend::CUDA> lhs({ 1.0f, 2.0f });
	const vext::Tensor<float, vext::Backend::CUDA> rhs({ 3.0f, 4.0f });
	const auto                                     result = vext::binary<vext::Op::ADD>(lhs, rhs);

	return result.length() == 2 ? 0 : 1;
}
