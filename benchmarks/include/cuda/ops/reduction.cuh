#ifndef __VEXT_BENCHMARKS_CUDA_REDUCTION_CUH__
#define __VEXT_BENCHMARKS_CUDA_REDUCTION_CUH__

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

struct SquareFunctor
{
	__host__ __device__ float
	operator()(const float value) const
	{
		return value * value;
	}
};

struct DeviceSquaredDifferenceFunctor
{
	const float* mean;

	__host__ __device__ float
	operator()(
		const float value) const
	{
		const float difference = value - *mean;
		return difference * difference;
	}
};

template <::vext::Op Kp>
struct ReductionFinalizeFunctor
{
	float scale;

	__host__ __device__ float
	operator()(
		const float value) const
	{
		if constexpr(Kp == ::vext::Op::MEAN || Kp == ::vext::Op::VAR)
			{
				return value * scale;
			}
		else if constexpr(Kp == ::vext::Op::STD)
			{
				return sqrtf(value * scale);
			}
		else if constexpr(Kp == ::vext::Op::L2_NORM)
			{
				return sqrtf(value);
			}
		else
			{
				return value;
			}
	}
};

template <::vext::Op Kp>
void
run_cub_reduction(
	const ::vext::benchmarks::cuda::DeviceBuffer<float>&  values,
	::vext::benchmarks::cuda::DeviceBuffer<float>&        mean,
	::vext::benchmarks::cuda::DeviceBuffer<float>&        out,
	::vext::benchmarks::cuda::DeviceBuffer<std::uint8_t>& workspace,
	std::uint64_t&                                        workspace_size,
	const std::int32_t                                    size)
{
	thrust::device_ptr<const float> first = thrust::device_pointer_cast(values.data());

	if constexpr(Kp == ::vext::Op::SUM || Kp == ::vext::Op::MEAN)
		{
			{
				const cudaError_t status = cub::DeviceReduce::Sum(
					workspace.data(),
					workspace_size,
					first,
					out.data(),
					size);
				::vext::benchmarks::cuda::check_cuda(status, "Running CUB sum reduction");
			}

			if constexpr(Kp == ::vext::Op::MEAN)
				{
					{
						const cudaError_t status = cub::DeviceTransform::Transform(
							out.data(),
							out.data(),
							1,
							ReductionFinalizeFunctor<Kp>{ 1.0f / size });
						::vext::benchmarks::cuda::check_cuda(status, "Finalizing CUB mean reduction");
					}
				}
		}
	else if constexpr(Kp == ::vext::Op::PROD)
		{
			{
				const cudaError_t status = cub::DeviceReduce::Reduce(
					workspace.data(),
					workspace_size,
					first,
					out.data(),
					size,
					thrust::multiplies<float>{},
					1.0f);
				::vext::benchmarks::cuda::check_cuda(status, "Running CUB product reduction");
			}
		}
	else if constexpr(Kp == ::vext::Op::MIN)
		{
			{
				const cudaError_t status = cub::DeviceReduce::Reduce(
					workspace.data(),
					workspace_size,
					first,
					out.data(),
					size,
					thrust::minimum<float>{},
					std::numeric_limits<float>::max());
				::vext::benchmarks::cuda::check_cuda(status, "Running CUB minimum reduction");
			}
		}
	else if constexpr(Kp == ::vext::Op::MAX)
		{
			{
				const cudaError_t status = cub::DeviceReduce::Reduce(
					workspace.data(),
					workspace_size,
					first,
					out.data(),
					size,
					thrust::maximum<float>{},
					std::numeric_limits<float>::lowest());
				::vext::benchmarks::cuda::check_cuda(status, "Running CUB maximum reduction");
			}
		}
	else if constexpr(Kp == ::vext::Op::L2_NORM)
		{
			thrust::transform_iterator<SquareFunctor, decltype(first)> squares = thrust::make_transform_iterator(first, SquareFunctor{});

			{
				const cudaError_t status = cub::DeviceReduce::Sum(
					workspace.data(),
					workspace_size,
					squares,
					out.data(),
					size);
				::vext::benchmarks::cuda::check_cuda(status, "Running CUB L2 reduction");
			}

			{
				const cudaError_t status = cub::DeviceTransform::Transform(
					out.data(),
					out.data(),
					1,
					ReductionFinalizeFunctor<Kp>{ 1.0f });
				::vext::benchmarks::cuda::check_cuda(status, "Finalizing CUB L2 reduction");
			}
		}
	else
		{
			{
				const cudaError_t status = cub::DeviceReduce::Sum(
					workspace.data(),
					workspace_size,
					first,
					mean.data(),
					size);
				::vext::benchmarks::cuda::check_cuda(status, "Running CUB variance mean reduction");
			}

			{
				const cudaError_t status = cub::DeviceTransform::Transform(
					mean.data(),
					mean.data(),
					1,
					ReductionFinalizeFunctor<::vext::Op::MEAN>{ 1.0f / size });
				::vext::benchmarks::cuda::check_cuda(status, "Finalizing CUB variance mean");
			}

			thrust::transform_iterator<DeviceSquaredDifferenceFunctor, decltype(first)> differences = thrust::make_transform_iterator(first, DeviceSquaredDifferenceFunctor{ mean.data() });

			{
				const cudaError_t status = cub::DeviceReduce::Sum(
					workspace.data(),
					workspace_size,
					differences,
					out.data(),
					size);
				::vext::benchmarks::cuda::check_cuda(status, "Running CUB squared-difference reduction");
			}

			{
				const cudaError_t status = cub::DeviceTransform::Transform(
					out.data(),
					out.data(),
					1,
					ReductionFinalizeFunctor<Kp>{ 1.0f / size });
				::vext::benchmarks::cuda::check_cuda(status, "Finalizing CUB variance reduction");
			}
		}
}

template <::vext::Op Kp>
void
BM_VextCudaReduction(
	benchmark::State& state)
{
	if(::vext::benchmarks::cuda::skip_without_cuda_device(state))
		{
			return;
		}

	const std::vector<std::uint32_t> shape = shape_from_state(state);
	const std::uint32_t              size  = shape_length(shape);

	::vext::Tensor<float, ::vext::Backend::CUDA> values(shape);
	::vext::Tensor<float, ::vext::Backend::CUDA> out(1);

	const ::vext::benchmarks::cuda::CudaEventTimer timer;

	values.set_from(make_reduction_values<Kp>(size));
	::vext::reduction<Kp>(values, ::vext::core::no_value_t{}, out);

	{
		const cudaError_t status = cudaDeviceSynchronize();
		::vext::benchmarks::cuda::check_cuda(status, "Warming up vext CUDA reduction");
	}

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			::vext::reduction<Kp>(values, ::vext::core::no_value_t{}, out);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <::vext::Op Kp>
void
BM_CubCudaReduction(
	benchmark::State& state)
{
	if(::vext::benchmarks::cuda::skip_without_cuda_device(state))
		{
			return;
		}

	const std::int32_t size = static_cast<std::int32_t>(shape_length(shape_from_state(state)));

	::vext::benchmarks::cuda::DeviceBuffer<float> values(size);
	::vext::benchmarks::cuda::DeviceBuffer<float> mean(1);
	::vext::benchmarks::cuda::DeviceBuffer<float> out(1);

	std::uint64_t                   workspace_size = 0;
	thrust::device_ptr<const float> first          = thrust::device_pointer_cast(values.data());

	if constexpr(Kp == ::vext::Op::PROD)
		{
			{
				const cudaError_t status = cub::DeviceReduce::Reduce(
					nullptr,
					workspace_size,
					first,
					out.data(),
					size,
					thrust::multiplies<float>{},
					1.0f);
				::vext::benchmarks::cuda::check_cuda(status, "Querying CUB product workspace");
			}
		}
	else if constexpr(Kp == ::vext::Op::MIN)
		{
			{
				const cudaError_t status = cub::DeviceReduce::Reduce(
					nullptr,
					workspace_size,
					first,
					out.data(),
					size,
					thrust::minimum<float>{},
					std::numeric_limits<float>::max());
				::vext::benchmarks::cuda::check_cuda(status, "Querying CUB minimum workspace");
			}
		}
	else if constexpr(Kp == ::vext::Op::MAX)
		{
			{
				const cudaError_t status = cub::DeviceReduce::Reduce(
					nullptr,
					workspace_size,
					first,
					out.data(),
					size,
					thrust::maximum<float>{},
					std::numeric_limits<float>::lowest());
				::vext::benchmarks::cuda::check_cuda(status, "Querying CUB maximum workspace");
			}
		}
	else if constexpr(Kp == ::vext::Op::L2_NORM)
		{
			thrust::transform_iterator<SquareFunctor, decltype(first)> squares = thrust::make_transform_iterator(first, SquareFunctor{});
			{
				const cudaError_t status = cub::DeviceReduce::Sum(
					nullptr,
					workspace_size,
					squares,
					out.data(),
					size);
				::vext::benchmarks::cuda::check_cuda(status, "Querying CUB L2 workspace");
			}
		}
	else
		{
			{
				const cudaError_t status = cub::DeviceReduce::Sum(
					nullptr,
					workspace_size,
					first,
					out.data(),
					size);
				::vext::benchmarks::cuda::check_cuda(status, "Querying CUB sum workspace");
			}

			if constexpr(Kp == ::vext::Op::STD || Kp == ::vext::Op::VAR)
				{
					std::uint64_t                                                               differences_workspace_size = 0;
					thrust::transform_iterator<DeviceSquaredDifferenceFunctor, decltype(first)> differences                = thrust::make_transform_iterator(first, DeviceSquaredDifferenceFunctor{ mean.data() });
					{
						const cudaError_t status = cub::DeviceReduce::Sum(
							nullptr,
							differences_workspace_size,
							differences,
							out.data(),
							size);
						::vext::benchmarks::cuda::check_cuda(status, "Querying CUB squared-difference workspace");
					}

					workspace_size = std::max(workspace_size, differences_workspace_size);
				}
		}

	::vext::benchmarks::cuda::DeviceBuffer<std::uint8_t> workspace(std::max<std::uint64_t>(workspace_size, 1));

	const ::vext::benchmarks::cuda::CudaEventTimer timer;

	values.fill_from_host(make_reduction_values<Kp>(size));
	run_cub_reduction<Kp>(values, mean, out, workspace, workspace_size, size);

	{
		const cudaError_t status = cudaDeviceSynchronize();
		::vext::benchmarks::cuda::check_cuda(status, "Warming up CUB CUDA reduction");
	}

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			run_cub_reduction<Kp>(values, mean, out, workspace, workspace_size, size);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * size);
}

struct RowSquaredDifferenceFunctor
{
	const float* values;
	const float* means;
	std::int32_t cols;

	__host__ __device__ float
	operator()(
		const std::int32_t index) const
	{
		const float difference = values[index] - means[index / cols];
		return difference * difference;
	}
};

template <bool SquareRoot>
struct ScaleFunctor
{
	float scale;

	__host__ __device__ float
	operator()(
		const float value) const
	{
		if constexpr(SquareRoot)
			{
				return sqrtf(value * scale);
			}
		else
			{
				return value * scale;
			}
	}
};

template <::vext::Op Kp>
void
run_cub_axis_reduction(
	const ::vext::benchmarks::cuda::DeviceBuffer<float>&        values,
	::vext::benchmarks::cuda::DeviceBuffer<float>&              means,
	::vext::benchmarks::cuda::DeviceBuffer<float>&              out,
	const ::vext::benchmarks::cuda::DeviceBuffer<std::int32_t>& offsets,
	::vext::benchmarks::cuda::DeviceBuffer<std::uint8_t>&       workspace,
	std::uint64_t&                                              workspace_size,
	const std::int32_t                                          rows,
	const std::int32_t                                          cols)
{
	thrust::counting_iterator<std::int32_t> indices = thrust::make_counting_iterator<std::int32_t>(0);
	thrust::device_ptr<const float>         first   = thrust::device_pointer_cast(values.data());

	if constexpr(Kp == ::vext::Op::SUM || Kp == ::vext::Op::MEAN)
		{
			{
				const cudaError_t status = cub::DeviceSegmentedReduce::Reduce(
					workspace.data(),
					workspace_size,
					first,
					out.data(),
					rows,
					offsets.data(),
					offsets.data() + 1,
					thrust::plus<float>{},
					0.0f);
				::vext::benchmarks::cuda::check_cuda(status, "Running CUB segmented sum reduction");
			}

			if constexpr(Kp == ::vext::Op::MEAN)
				{
					{
						const cudaError_t status = cub::DeviceTransform::Transform(
							out.data(),
							out.data(),
							rows,
							ScaleFunctor<false>{ 1.0f / cols });
						::vext::benchmarks::cuda::check_cuda(status, "Finalizing CUB segmented mean reduction");
					}
				}
		}
	else if constexpr(Kp == ::vext::Op::PROD)
		{
			{
				const cudaError_t status = cub::DeviceSegmentedReduce::Reduce(
					workspace.data(),
					workspace_size,
					first,
					out.data(),
					rows,
					offsets.data(),
					offsets.data() + 1,
					thrust::multiplies<float>{},
					1.0f);
				::vext::benchmarks::cuda::check_cuda(status, "Running CUB segmented product reduction");
			}
		}
	else if constexpr(Kp == ::vext::Op::MIN)
		{
			{
				const cudaError_t status = cub::DeviceSegmentedReduce::Reduce(
					workspace.data(),
					workspace_size,
					first,
					out.data(),
					rows,
					offsets.data(),
					offsets.data() + 1,
					thrust::minimum<float>{},
					std::numeric_limits<float>::max());
				::vext::benchmarks::cuda::check_cuda(status, "Running CUB segmented minimum reduction");
			}
		}
	else if constexpr(Kp == ::vext::Op::MAX)
		{
			{
				const cudaError_t status = cub::DeviceSegmentedReduce::Reduce(
					workspace.data(),
					workspace_size,
					first,
					out.data(),
					rows,
					offsets.data(),
					offsets.data() + 1,
					thrust::maximum<float>{},
					std::numeric_limits<float>::lowest());
				::vext::benchmarks::cuda::check_cuda(status, "Running CUB segmented maximum reduction");
			}
		}
	else if constexpr(Kp == ::vext::Op::L2_NORM)
		{
			thrust::transform_iterator<SquareFunctor, decltype(first)> squares = thrust::make_transform_iterator(first, SquareFunctor{});

			{
				const cudaError_t status = cub::DeviceSegmentedReduce::Reduce(
					workspace.data(),
					workspace_size,
					squares,
					out.data(),
					rows,
					offsets.data(),
					offsets.data() + 1,
					thrust::plus<float>{},
					0.0f);
				::vext::benchmarks::cuda::check_cuda(status, "Running CUB segmented L2 reduction");
			}

			{
				const cudaError_t status = cub::DeviceTransform::Transform(
					out.data(),
					out.data(),
					rows,
					ScaleFunctor<true>{ 1.0f });
				::vext::benchmarks::cuda::check_cuda(status, "Finalizing CUB segmented L2 reduction");
			}
		}
	else
		{
			{
				const cudaError_t status = cub::DeviceSegmentedReduce::Reduce(
					workspace.data(),
					workspace_size,
					first,
					means.data(),
					rows,
					offsets.data(),
					offsets.data() + 1,
					thrust::plus<float>{},
					0.0f);
				::vext::benchmarks::cuda::check_cuda(status, "Running CUB segmented variance means");
			}

			{
				const cudaError_t status = cub::DeviceTransform::Transform(
					means.data(),
					means.data(),
					rows,
					ScaleFunctor<false>{ 1.0f / cols });
				::vext::benchmarks::cuda::check_cuda(status, "Finalizing CUB segmented variance means");
			}

			thrust::transform_iterator<RowSquaredDifferenceFunctor, decltype(indices)> differences = thrust::make_transform_iterator(indices, RowSquaredDifferenceFunctor{ values.data(), means.data(), cols });
			{
				const cudaError_t status = cub::DeviceSegmentedReduce::Reduce(
					workspace.data(),
					workspace_size,
					differences,
					out.data(),
					rows,
					offsets.data(),
					offsets.data() + 1,
					thrust::plus<float>{},
					0.0f);
				::vext::benchmarks::cuda::check_cuda(status, "Running CUB segmented squared-difference reduction");
			}

			if constexpr(Kp == ::vext::Op::STD)
				{
					{
						const cudaError_t status = cub::DeviceTransform::Transform(
							out.data(),
							out.data(),
							rows,
							ScaleFunctor<true>{ 1.0f / cols });
						::vext::benchmarks::cuda::check_cuda(status, "Finalizing CUB segmented standard deviation");
					}
				}
			else
				{
					{
						const cudaError_t status = cub::DeviceTransform::Transform(
							out.data(),
							out.data(),
							rows,
							ScaleFunctor<false>{ 1.0f / cols });
						::vext::benchmarks::cuda::check_cuda(status, "Finalizing CUB segmented variance");
					}
				}
		}
}

template <::vext::Op Kp>
void
BM_VextCudaAxisReduction(
	benchmark::State& state)
{
	if(::vext::benchmarks::cuda::skip_without_cuda_device(state))
		{
			return;
		}

	const std::vector<std::uint32_t> shape = shape_from_state(state);
	const std::uint32_t              size  = shape_length(shape);
	const std::uint32_t              cols  = shape.back();
	std::vector<std::uint32_t>       output_shape(shape.begin(), shape.end() - 1);

	if(output_shape.empty())
		output_shape.push_back(1);

	::vext::Tensor<float, ::vext::Backend::CUDA> values(shape);
	::vext::Tensor<float, ::vext::Backend::CUDA> out(output_shape);

	const ::vext::benchmarks::cuda::CudaEventTimer timer;

	values.set_from(make_reduction_values<Kp>(size));
	::vext::reduction<Kp>(values, ::vext::axes({ static_cast<std::int32_t>(shape.size() - 1) }), out);

	{
		const cudaError_t status = cudaDeviceSynchronize();
		::vext::benchmarks::cuda::check_cuda(status, "Warming up vext CUDA axis reduction");
	}

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			::vext::reduction<Kp>(values, ::vext::axes({ static_cast<std::int32_t>(shape.size() - 1) }), out);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <::vext::Op Kp>
void
BM_CubCudaAxisReduction(
	benchmark::State& state)
{
	if(::vext::benchmarks::cuda::skip_without_cuda_device(state))
		{
			return;
		}

	const std::vector<std::uint32_t> shape = shape_from_state(state);
	const std::int32_t               cols  = static_cast<std::int32_t>(shape.back());
	const std::int32_t               rows  = static_cast<std::int32_t>(shape_length(shape) / shape.back());

	::vext::benchmarks::cuda::DeviceBuffer<float>        values(static_cast<std::int64_t>(rows) * cols);
	::vext::benchmarks::cuda::DeviceBuffer<float>        means(rows);
	::vext::benchmarks::cuda::DeviceBuffer<float>        out(rows);
	::vext::benchmarks::cuda::DeviceBuffer<std::int32_t> offsets(rows + 1);

	std::vector<std::int32_t> host_offsets(static_cast<std::uint64_t>(rows) + 1);

	for(std::int32_t row = 0; row <= rows; ++row)
		{
			host_offsets[static_cast<std::uint64_t>(row)] = row * cols;
		}

	offsets.fill_from_host(host_offsets);

	std::uint64_t                   workspace_size = 0;
	thrust::device_ptr<const float> first          = thrust::device_pointer_cast(values.data());

	if constexpr(Kp == ::vext::Op::PROD)
		{
			{
				const cudaError_t status = cub::DeviceSegmentedReduce::Reduce(
					nullptr,
					workspace_size,
					first,
					out.data(),
					rows,
					offsets.data(),
					offsets.data() + 1,
					thrust::multiplies<float>{},
					1.0f);
				::vext::benchmarks::cuda::check_cuda(status, "Querying CUB segmented product workspace");
			}
		}
	else if constexpr(Kp == ::vext::Op::MIN)
		{
			{
				const cudaError_t status = cub::DeviceSegmentedReduce::Reduce(
					nullptr,
					workspace_size,
					first,
					out.data(),
					rows,
					offsets.data(),
					offsets.data() + 1,
					thrust::minimum<float>{},
					std::numeric_limits<float>::max());
				::vext::benchmarks::cuda::check_cuda(status, "Querying CUB segmented minimum workspace");
			}
		}
	else if constexpr(Kp == ::vext::Op::MAX)
		{
			{
				const cudaError_t status = cub::DeviceSegmentedReduce::Reduce(
					nullptr,
					workspace_size,
					first,
					out.data(),
					rows,
					offsets.data(),
					offsets.data() + 1,
					thrust::maximum<float>{},
					std::numeric_limits<float>::lowest());
				::vext::benchmarks::cuda::check_cuda(status, "Querying CUB segmented maximum workspace");
			}
		}
	else if constexpr(Kp == ::vext::Op::L2_NORM)
		{
			thrust::transform_iterator<SquareFunctor, decltype(first)> squares = thrust::make_transform_iterator(first, SquareFunctor{});

			{
				const cudaError_t status = cub::DeviceSegmentedReduce::Reduce(
					nullptr,
					workspace_size,
					squares,
					out.data(),
					rows,
					offsets.data(),
					offsets.data() + 1,
					thrust::plus<float>{},
					0.0f);
				::vext::benchmarks::cuda::check_cuda(status, "Querying CUB segmented L2 workspace");
			}
		}
	else
		{
			{
				const cudaError_t status = cub::DeviceSegmentedReduce::Reduce(
					nullptr,
					workspace_size,
					first,
					out.data(),
					rows,
					offsets.data(),
					offsets.data() + 1,
					thrust::plus<float>{},
					0.0f);
				::vext::benchmarks::cuda::check_cuda(status, "Querying CUB segmented sum workspace");
			}

			if constexpr(Kp == ::vext::Op::STD || Kp == ::vext::Op::VAR)
				{
					std::uint64_t                                                              differences_workspace_size = 0;
					thrust::counting_iterator<std::int32_t>                                    indices                    = thrust::make_counting_iterator<std::int32_t>(0);
					thrust::transform_iterator<RowSquaredDifferenceFunctor, decltype(indices)> differences                = thrust::make_transform_iterator(indices, RowSquaredDifferenceFunctor{ values.data(), means.data(), cols });

					{
						const cudaError_t status = cub::DeviceSegmentedReduce::Reduce(
							nullptr,
							differences_workspace_size,
							differences,
							out.data(),
							rows,
							offsets.data(),
							offsets.data() + 1,
							thrust::plus<float>{},
							0.0f);
						::vext::benchmarks::cuda::check_cuda(status, "Querying CUB segmented squared-difference workspace");
					}

					workspace_size = std::max(workspace_size, differences_workspace_size);
				}
		}

	::vext::benchmarks::cuda::DeviceBuffer<std::uint8_t> workspace(std::max<std::uint64_t>(workspace_size, 1));

	const ::vext::benchmarks::cuda::CudaEventTimer timer;

	values.fill_from_host(make_reduction_values<Kp>(static_cast<std::int64_t>(rows) * cols));
	run_cub_axis_reduction<Kp>(values, means, out, offsets, workspace, workspace_size, rows, cols);

	{
		const cudaError_t status = cudaDeviceSynchronize();
		::vext::benchmarks::cuda::check_cuda(status, "Warming up CUB CUDA axis reduction");
	}

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			run_cub_axis_reduction<Kp>(values, means, out, offsets, workspace, workspace_size, rows, cols);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * rows * cols);
}

} // namespace vext::benchmarks::cuda::ops

#endif
