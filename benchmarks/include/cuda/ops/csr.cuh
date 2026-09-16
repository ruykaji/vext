#ifndef __VEXT_BENCHMARKS_CUDA_CSR_CUH__
#define __VEXT_BENCHMARKS_CUDA_CSR_CUH__

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_transform.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <vext/ops.hpp>
#include <vext/tensor.hpp>

#include <cuda/buffer.cuh>
#include <cuda/common.cuh>
#include <cuda/timing.cuh>

namespace vext::benchmarks::cuda::ops
{

struct CsrData
{
	std::vector<float>         values;
	std::vector<std::uint32_t> head;
	std::vector<std::uint32_t> tail;
};

CsrData
make_csr_data(
	const std::int64_t CSR_ROWS,
	const std::int64_t CSR_DEGREE)
{
	CsrData data;
	data.values = make_values(CSR_ROWS * CSR_DEGREE, true);
	data.head.resize(CSR_ROWS + 1);
	data.tail.resize(CSR_ROWS * CSR_DEGREE);

	for(std::int64_t row = 0; row < CSR_ROWS; ++row)
		{
			data.head[row] = static_cast<std::uint32_t>(row * CSR_DEGREE);

			for(std::int64_t entry = 0; entry < CSR_DEGREE; ++entry)
				{
					data.tail[row * CSR_DEGREE + entry] = static_cast<std::uint32_t>((row + entry * 17) % CSR_ROWS);
				}
		}

	data.head.back() = static_cast<std::uint32_t>(CSR_ROWS * CSR_DEGREE);
	return data;
}

template <::vext::Op Kp>
std::vector<float>
make_cusparse_values(
	const CsrData& data,
	const bool     use_values,
	const std::int64_t CSR_DEGREE)
{
	std::vector<float> values(data.values.size(), 1.0f);

	if(use_values)
		{
			values = data.values;
		}

	if constexpr(Kp == ::vext::Op::MEAN)
		{
			for(float& value : values)
				{
					value /= static_cast<float>(CSR_DEGREE);
				}
		}

	return values;
}

template <::vext::Op Kp>
void
BM_VextCudaCsrScatter(
	benchmark::State& state)
{
	const std::int64_t CSR_ROWS = state.range(0), CSR_DEGREE = state.range(1), CSR_FEATURES = state.range(2);
	if(::vext::benchmarks::cuda::skip_without_cuda_device(state))
		{
			return;
		}

	const CsrData            data       = make_csr_data(CSR_ROWS, CSR_DEGREE);
	const std::vector<float> src_values = make_values(CSR_ROWS * CSR_FEATURES);

	::vext::Tensor<float, ::vext::Backend::CUDA>         src(CSR_ROWS, CSR_FEATURES);
	::vext::Tensor<std::uint32_t, ::vext::Backend::CUDA> head(CSR_ROWS + 1);
	::vext::Tensor<std::uint32_t, ::vext::Backend::CUDA> tail(CSR_ROWS * CSR_DEGREE);
	::vext::Tensor<float, ::vext::Backend::CUDA>         out(CSR_ROWS, CSR_FEATURES);

	const ::vext::benchmarks::cuda::CudaEventTimer timer;

	src.set_from(src_values);
	head.set_from(data.head);
	tail.set_from(data.tail);
	::vext::csr_scatter<Kp>(src, head, tail, out);

	{
		const cudaError_t status = cudaDeviceSynchronize();
		::vext::benchmarks::cuda::check_cuda(status, "Warming up vext CUDA CSR scatter");
	}

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			::vext::csr_scatter<Kp>(src, head, tail, out);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * CSR_ROWS * CSR_DEGREE * CSR_FEATURES);
}

template <::vext::Op Kp>
void
BM_CusparseCudaCsrScatter(
	benchmark::State& state)
{
	const std::int64_t CSR_ROWS = state.range(0), CSR_DEGREE = state.range(1), CSR_FEATURES = state.range(2);
	if(::vext::benchmarks::cuda::skip_without_cuda_device(state))
		{
			return;
		}

	const CsrData data = make_csr_data(CSR_ROWS, CSR_DEGREE);

	::vext::benchmarks::cuda::DeviceBuffer<std::uint32_t> head(CSR_ROWS + 1);
	::vext::benchmarks::cuda::DeviceBuffer<std::uint32_t> tail(CSR_ROWS * CSR_DEGREE);
	::vext::benchmarks::cuda::DeviceBuffer<float>         values(CSR_ROWS * CSR_DEGREE);
	::vext::benchmarks::cuda::DeviceBuffer<float>         src(CSR_ROWS * CSR_FEATURES);
	::vext::benchmarks::cuda::DeviceBuffer<float>         out(CSR_ROWS * CSR_FEATURES);

	const ::vext::benchmarks::cuda::CusparseHandle handle;
	const ::vext::benchmarks::cuda::CudaEventTimer timer;

	head.fill_from_host(data.head);
	tail.fill_from_host(data.tail);
	values.fill_from_host(make_cusparse_values<Kp>(data, false, CSR_DEGREE));
	src.fill_from_host(make_values(CSR_ROWS * CSR_FEATURES));

	cusparseSpMatDescr_t matrix    = nullptr;
	cusparseDnMatDescr_t dense_src = nullptr;
	cusparseDnMatDescr_t dense_out = nullptr;

	{
		const cusparseStatus_t status = cusparseCreateCsr(
			&matrix,
			CSR_ROWS,
			CSR_ROWS,
			CSR_ROWS * CSR_DEGREE,
			head.data(),
			tail.data(),
			values.data(),
			CUSPARSE_INDEX_32I,
			CUSPARSE_INDEX_32I,
			CUSPARSE_INDEX_BASE_ZERO,
			CUDA_R_32F);
		::vext::benchmarks::cuda::check_cusparse(status, "Creating the cuSPARSE CSR scatter matrix");
	}

	{
		const cusparseStatus_t status = cusparseCreateDnMat(
			&dense_src,
			CSR_ROWS,
			CSR_FEATURES,
			CSR_FEATURES,
			src.data(),
			CUDA_R_32F,
			CUSPARSE_ORDER_ROW);
		::vext::benchmarks::cuda::check_cusparse(status, "Creating the cuSPARSE scatter source matrix");
	}

	{
		const cusparseStatus_t status = cusparseCreateDnMat(
			&dense_out,
			CSR_ROWS,
			CSR_FEATURES,
			CSR_FEATURES,
			out.data(),
			CUDA_R_32F,
			CUSPARSE_ORDER_ROW);
		::vext::benchmarks::cuda::check_cusparse(status, "Creating the cuSPARSE scatter output matrix");
	}

	const float alpha = 1.0f;
	const float beta  = 0.0f;

	std::uint64_t workspace_size = 0;

	{
		const cusparseStatus_t status = cusparseSpMM_bufferSize(
			handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha,
			matrix,
			dense_src,
			&beta,
			dense_out,
			CUDA_R_32F,
			CUSPARSE_SPMM_ALG_DEFAULT,
			&workspace_size);
		::vext::benchmarks::cuda::check_cusparse(status, "Querying cuSPARSE SpMM workspace size");
	}

	::vext::benchmarks::cuda::DeviceBuffer<std::uint8_t> workspace(std::max<std::uint64_t>(workspace_size, 1));

	{
		const cusparseStatus_t status = cusparseSpMM_preprocess(
			handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha,
			matrix,
			dense_src,
			&beta,
			dense_out,
			CUDA_R_32F,
			CUSPARSE_SPMM_ALG_DEFAULT,
			workspace.data());
		::vext::benchmarks::cuda::check_cusparse(status, "Preprocessing cuSPARSE CSR scatter through SpMM");
	}

	{
		const cusparseStatus_t status = cusparseSpMM(
			handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha,
			matrix,
			dense_src,
			&beta,
			dense_out,
			CUDA_R_32F,
			CUSPARSE_SPMM_ALG_DEFAULT,
			workspace.data());
		::vext::benchmarks::cuda::check_cusparse(status, "Warming up cuSPARSE CSR scatter through SpMM");
	}

	{
		const cudaError_t status = cudaDeviceSynchronize();
		::vext::benchmarks::cuda::check_cuda(status, "Synchronizing cuSPARSE SpMM warmup");
	}

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();

			{
				const cusparseStatus_t status = cusparseSpMM(
					handle,
					CUSPARSE_OPERATION_NON_TRANSPOSE,
					CUSPARSE_OPERATION_NON_TRANSPOSE,
					&alpha,
					matrix,
					dense_src,
					&beta,
					dense_out,
					CUDA_R_32F,
					CUSPARSE_SPMM_ALG_DEFAULT,
					workspace.data());
				::vext::benchmarks::cuda::check_cusparse(status, "Running cuSPARSE CSR scatter through SpMM");
			}

			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	::vext::benchmarks::cuda::check_cusparse(cusparseDestroyDnMat(dense_out), "Destroying the cuSPARSE scatter output matrix");
	::vext::benchmarks::cuda::check_cusparse(cusparseDestroyDnMat(dense_src), "Destroying the cuSPARSE scatter source matrix");
	::vext::benchmarks::cuda::check_cusparse(cusparseDestroySpMat(matrix), "Destroying the cuSPARSE CSR scatter matrix");

	state.SetItemsProcessed(state.iterations() * CSR_ROWS * CSR_DEGREE * CSR_FEATURES);
}

template <::vext::Op Kp>
void
BM_VextCudaCsrSpmv(
	benchmark::State& state)
{
	const std::int64_t CSR_ROWS = state.range(0), CSR_DEGREE = state.range(1);
	if(::vext::benchmarks::cuda::skip_without_cuda_device(state))
		{
			return;
		}

	const CsrData data = make_csr_data(CSR_ROWS, CSR_DEGREE);

	::vext::Tensor<float, ::vext::Backend::CUDA>         values(CSR_ROWS * CSR_DEGREE);
	::vext::Tensor<std::uint32_t, ::vext::Backend::CUDA> head(CSR_ROWS + 1);
	::vext::Tensor<std::uint32_t, ::vext::Backend::CUDA> tail(CSR_ROWS * CSR_DEGREE);
	::vext::Tensor<float, ::vext::Backend::CUDA>         x(CSR_ROWS);
	::vext::Tensor<float, ::vext::Backend::CUDA>         out(CSR_ROWS);

	const ::vext::benchmarks::cuda::CudaEventTimer timer;

	values.set_from(data.values);
	head.set_from(data.head);
	tail.set_from(data.tail);
	x.set_from(make_values(CSR_ROWS));
	::vext::csr_spmv<Kp>(values, head, tail, x, out);

	{
		const cudaError_t status = cudaDeviceSynchronize();
		::vext::benchmarks::cuda::check_cuda(status, "Warming up vext CUDA CSR SpMV");
	}

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			::vext::csr_spmv<Kp>(values, head, tail, x, out);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * CSR_ROWS * CSR_DEGREE);
}

template <::vext::Op Kp>
void
BM_CusparseCudaCsrSpmv(
	benchmark::State& state)
{
	const std::int64_t CSR_ROWS = state.range(0), CSR_DEGREE = state.range(1);
	if(::vext::benchmarks::cuda::skip_without_cuda_device(state))
		{
			return;
		}

	const CsrData data = make_csr_data(CSR_ROWS, CSR_DEGREE);

	::vext::benchmarks::cuda::DeviceBuffer<std::uint32_t> head(CSR_ROWS + 1);
	::vext::benchmarks::cuda::DeviceBuffer<std::uint32_t> tail(CSR_ROWS * CSR_DEGREE);
	::vext::benchmarks::cuda::DeviceBuffer<float>         values(CSR_ROWS * CSR_DEGREE);
	::vext::benchmarks::cuda::DeviceBuffer<float>         x(CSR_ROWS);
	::vext::benchmarks::cuda::DeviceBuffer<float>         out(CSR_ROWS);

	const ::vext::benchmarks::cuda::CusparseHandle handle;
	const ::vext::benchmarks::cuda::CudaEventTimer timer;

	head.fill_from_host(data.head);
	tail.fill_from_host(data.tail);
	values.fill_from_host(make_cusparse_values<Kp>(data, true, CSR_DEGREE));
	x.fill_from_host(make_values(CSR_ROWS));

	cusparseSpMatDescr_t matrix    = nullptr;
	cusparseDnVecDescr_t dense_x   = nullptr;
	cusparseDnVecDescr_t dense_out = nullptr;

	{
		const cusparseStatus_t status = cusparseCreateCsr(
			&matrix,
			CSR_ROWS,
			CSR_ROWS,
			CSR_ROWS * CSR_DEGREE,
			head.data(),
			tail.data(),
			values.data(),
			CUSPARSE_INDEX_32I,
			CUSPARSE_INDEX_32I,
			CUSPARSE_INDEX_BASE_ZERO,
			CUDA_R_32F);
		::vext::benchmarks::cuda::check_cusparse(status, "Creating the cuSPARSE CSR matrix");
	}

	{
		const cusparseStatus_t status = cusparseCreateDnVec(
			&dense_x,
			CSR_ROWS,
			x.data(),
			CUDA_R_32F);
		::vext::benchmarks::cuda::check_cusparse(status, "Creating the cuSPARSE input vector");
	}

	{
		const cusparseStatus_t status = cusparseCreateDnVec(
			&dense_out,
			CSR_ROWS,
			out.data(),
			CUDA_R_32F);
		::vext::benchmarks::cuda::check_cusparse(status, "Creating the cuSPARSE output vector");
	}

	const float alpha = 1.0f;
	const float beta  = 0.0f;

	std::uint64_t               workspace_size = 0;
	constexpr cusparseSpMVAlg_t SPMV_ALGORITHM = CUSPARSE_SPMV_CSR_ALG1;

	{
		const cusparseStatus_t status = cusparseSpMV_bufferSize(
			handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha,
			matrix,
			dense_x,
			&beta,
			dense_out,
			CUDA_R_32F,
			SPMV_ALGORITHM,
			&workspace_size);
		::vext::benchmarks::cuda::check_cusparse(status, "Querying cuSPARSE SpMV workspace size");
	}

	::vext::benchmarks::cuda::DeviceBuffer<std::uint8_t> workspace(std::max<std::uint64_t>(workspace_size, 1));

	{
		const cusparseStatus_t status = cusparseSpMV_preprocess(
			handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha,
			matrix,
			dense_x,
			&beta,
			dense_out,
			CUDA_R_32F,
			SPMV_ALGORITHM,
			workspace.data());
		::vext::benchmarks::cuda::check_cusparse(status, "Preprocessing cuSPARSE SpMV");
	}

	{
		const cusparseStatus_t status = cusparseSpMV(
			handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha,
			matrix,
			dense_x,
			&beta,
			dense_out,
			CUDA_R_32F,
			SPMV_ALGORITHM,
			workspace.data());
		::vext::benchmarks::cuda::check_cusparse(status, "Warming up cuSPARSE SpMV");
	}

	{
		const cudaError_t status = cudaDeviceSynchronize();
		::vext::benchmarks::cuda::check_cuda(status, "Synchronizing cuSPARSE SpMV warmup");
	}

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			{
				const cusparseStatus_t status = cusparseSpMV(
					handle,
					CUSPARSE_OPERATION_NON_TRANSPOSE,
					&alpha,
					matrix,
					dense_x,
					&beta,
					dense_out,
					CUDA_R_32F,
					SPMV_ALGORITHM,
					workspace.data());
				::vext::benchmarks::cuda::check_cusparse(status, "Running cuSPARSE SpMV");
			}

			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	::vext::benchmarks::cuda::check_cusparse(cusparseDestroyDnVec(dense_out), "Destroying the cuSPARSE output vector");
	::vext::benchmarks::cuda::check_cusparse(cusparseDestroyDnVec(dense_x), "Destroying the cuSPARSE input vector");
	::vext::benchmarks::cuda::check_cusparse(cusparseDestroySpMat(matrix), "Destroying the cuSPARSE CSR matrix");

	state.SetItemsProcessed(state.iterations() * CSR_ROWS * CSR_DEGREE);
}

} // namespace vext::benchmarks::cuda::ops

#endif
