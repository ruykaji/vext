#ifndef __VEXT_BENCHMARKS_CPU_CSR_HPP__
#define __VEXT_BENCHMARKS_CPU_CSR_HPP__

#include <benchmark/benchmark.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <vext/ops.hpp>
#include <vext/tensor.hpp>

#include <cpu/common.hpp>

namespace vext::benchmarks::cpu::ops
{

struct CsrData
{
	std::vector<float>         values;
	std::vector<std::uint32_t> head;
	std::vector<std::uint32_t> tail;
};

CsrData
make_csr_data(
	const std::int64_t rows,
	const std::int64_t degree)
{
	CsrData data;
	data.values = make_values(rows * degree, true);
	data.head.resize(rows + 1);
	data.tail.resize(rows * degree);

	for(std::int64_t row = 0; row < rows; ++row)
		{
			data.head[row] = static_cast<std::uint32_t>(row * degree);

			for(std::int64_t entry = 0; entry < degree; ++entry)
				{
					data.tail[row * degree + entry] = static_cast<std::uint32_t>((row + entry * 17) % rows);
				}
		}

	data.head.back() = static_cast<std::uint32_t>(rows * degree);
	return data;
}

template <::vext::Op Kp>
Eigen::SparseMatrix<float, Eigen::RowMajor>
make_eigen_sparse(
	const CsrData& data,
	const bool     use_values,
	const std::int64_t rows,
	const std::int64_t degree)
{
	std::vector<Eigen::Triplet<float>> entries;
	entries.reserve(data.tail.size());

	for(std::int64_t row = 0; row < rows; ++row)
		{
			for(std::int64_t entry = 0; entry < degree; ++entry)
				{
					const std::int64_t index = row * degree + entry;
					float              value = use_values ? data.values[index] : 1.0f;

					if constexpr(Kp == ::vext::Op::MEAN)
						{
						value /= static_cast<float>(degree);
						}

					entries.emplace_back(row, data.tail[index], value);
				}
		}

	Eigen::SparseMatrix<float, Eigen::RowMajor> matrix(rows, rows);
	matrix.setFromTriplets(entries.begin(), entries.end());
	return matrix;
}

template <::vext::Op Kp>
void
BM_VextCpuCsrScatter(
	benchmark::State& state)
{
	const std::int64_t rows = state.range(0), degree = state.range(1), features = state.range(2);
	const CsrData            data       = make_csr_data(rows, degree);
	const std::vector<float> src_values = make_values(rows * features);

	::vext::Tensor<float>         src(rows, features);
	::vext::Tensor<std::uint32_t> head(rows + 1);
	::vext::Tensor<std::uint32_t> tail(rows * degree);
	::vext::Tensor<float>         out(rows, features);
	src.set_from(src_values);
	head.set_from(data.head);
	tail.set_from(data.tail);

	for([[maybe_unused]] auto iteration : state)
		{
			::vext::csr_scatter<Kp>(src, head, tail, out);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * rows * degree * features);
}

template <::vext::Op Kp>
void
BM_EigenCpuCsrScatter(
	benchmark::State& state)
{
	const std::int64_t rows = state.range(0), degree = state.range(1), features = state.range(2);
	const CsrData                                     data       = make_csr_data(rows, degree);
	const std::vector<float>                          src_values = make_values(rows * features);
	const Eigen::SparseMatrix<float, Eigen::RowMajor> matrix     = make_eigen_sparse<Kp>(data, false, rows, degree);

	const Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> src(src_values.data(), rows, features);

	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out(rows, features);

	for([[maybe_unused]] auto iteration : state)
		{
			out.noalias() = matrix * src;
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * rows * degree * features);
}

template <::vext::Op Kp>
void
BM_VextCpuCsrSpmv(
	benchmark::State& state)
{
	const std::int64_t rows = state.range(0), degree = state.range(1);
	const CsrData            data     = make_csr_data(rows, degree);
	const std::vector<float> x_values = make_values(rows);

	::vext::Tensor<float>         values(rows * degree);
	::vext::Tensor<std::uint32_t> head(rows + 1);
	::vext::Tensor<std::uint32_t> tail(rows * degree);
	::vext::Tensor<float>         x(rows);
	::vext::Tensor<float>         out(rows);
	values.set_from(data.values);
	head.set_from(data.head);
	tail.set_from(data.tail);
	x.set_from(x_values);

	for([[maybe_unused]] auto iteration : state)
		{
			::vext::csr_spmv<Kp>(values, head, tail, x, out);
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * rows * degree);
}

template <::vext::Op Kp>
void
BM_EigenCpuCsrSpmv(
	benchmark::State& state)
{
	const std::int64_t rows = state.range(0), degree = state.range(1);
	const CsrData                                     data     = make_csr_data(rows, degree);
	const std::vector<float>                          x_values = make_values(rows);
	const Eigen::SparseMatrix<float, Eigen::RowMajor> matrix   = make_eigen_sparse<Kp>(data, true, rows, degree);

	const Eigen::Map<const Eigen::VectorXf> x(x_values.data(), rows);

	Eigen::VectorXf out(rows);

	for([[maybe_unused]] auto iteration : state)
		{
			out.noalias() = matrix * x;
			benchmark::DoNotOptimize(out.data());
			benchmark::ClobberMemory();
		}

	state.SetItemsProcessed(state.iterations() * rows * degree);
}

} // namespace vext::benchmarks::cpu::ops

#endif
