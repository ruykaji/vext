#include <benchmark/benchmark.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <vext/ops.hpp>
#include <vext/tensor.hpp>

namespace
{

constexpr std::int64_t VECTOR_SIZE   = 1 << 20;
constexpr std::int64_t MATRIX_ROWS   = 1024;
constexpr std::int64_t MATRIX_COLS   = 1024;
constexpr std::int64_t CSR_ROWS      = 4096;
constexpr std::int64_t CSR_DEGREE    = 16;
constexpr std::int64_t CSR_FEATURES  = 64;
constexpr float        UNARY_PARAM_A = 1.25f;
constexpr float        UNARY_PARAM_B = 2.0f;

void
check_cuda(
	const cudaError_t status,
	const char*       operation)
{
	if(status != cudaSuccess)
		{
			throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
		}
}

void
check_cublas(
	const cublasStatus_t status,
	const char*          operation)
{
	if(status != CUBLAS_STATUS_SUCCESS)
		{
			throw std::runtime_error(std::string(operation) + " failed with cuBLAS status " + std::to_string(status) + ".");
		}
}

void
check_cusparse(
	const cusparseStatus_t status,
	const char*            operation)
{
	if(status != CUSPARSE_STATUS_SUCCESS)
		{
			throw std::runtime_error(std::string(operation) + " failed with cuSPARSE status " + std::to_string(status) + ".");
		}
}

bool
skip_without_cuda_device(
	benchmark::State& state)
{
	std::int32_t count = 0;

	if(cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
		{
			state.SkipWithError("No CUDA-capable device is available.");
			return true;
		}

	return false;
}

class CudaEventTimer
{
public:
	CudaEventTimer()
	{
		check_cuda(cudaEventCreate(&__start), "Creating the start event");
		check_cuda(cudaEventCreate(&__stop), "Creating the stop event");
	}

	~CudaEventTimer()
	{
		cudaEventDestroy(__stop);
		cudaEventDestroy(__start);
	}

public:
	void
	start() const
	{
		check_cuda(cudaEventRecord(__start), "Recording the start event");
	}

	double
	stop_seconds() const
	{
		check_cuda(cudaEventRecord(__stop), "Recording the stop event");
		check_cuda(cudaEventSynchronize(__stop), "Synchronizing the stop event");

		float milliseconds = 0.0f;
		check_cuda(cudaEventElapsedTime(&milliseconds, __start, __stop), "Measuring elapsed CUDA time");
		return static_cast<double>(milliseconds) / 1000.0;
	}

private:
	cudaEvent_t __start = nullptr;
	cudaEvent_t __stop  = nullptr;
};

class CublasHandle
{
public:
	CublasHandle()
	{
		check_cublas(cublasCreate(&__handle), "Creating the cuBLAS handle");
	}

	~CublasHandle()
	{
		cublasDestroy(__handle);
	}

public:
	operator cublasHandle_t() const noexcept
	{
		return __handle;
	}

private:
	cublasHandle_t __handle = nullptr;
};

class CusparseHandle
{
public:
	CusparseHandle()
	{
		check_cusparse(cusparseCreate(&__handle), "Creating the cuSPARSE handle");
	}

	~CusparseHandle()
	{
		cusparseDestroy(__handle);
	}

public:
	operator cusparseHandle_t() const noexcept
	{
		return __handle;
	}

private:
	cusparseHandle_t __handle = nullptr;
};

template <typename Tp>
class DeviceBuffer
{
public:
	explicit DeviceBuffer(
		const std::size_t size)
		: __size(size)
	{
		check_cuda(cudaMalloc(&__data, __size * sizeof(Tp)), "Allocating a CUDA benchmark buffer");
	}

	~DeviceBuffer()
	{
		cudaFree(__data);
	}

	DeviceBuffer(
		const DeviceBuffer&) = delete;

	DeviceBuffer&
	operator=(
		const DeviceBuffer&) = delete;

public:
	Tp*
	data() noexcept
	{
		return __data;
	}

	const Tp*
	data() const noexcept
	{
		return __data;
	}

	void
	fill_from_host(
		const std::vector<Tp>& values)
	{
		check_cuda(cudaMemcpy(__data, values.data(), __size * sizeof(Tp), cudaMemcpyHostToDevice), "Copying benchmark data to CUDA");
	}

private:
	Tp*         __data = nullptr;
	std::size_t __size = 0;
};

std::vector<float>
make_values(
	const std::int64_t size,
	const bool         positive = false)
{
	std::vector<float> values(static_cast<std::size_t>(size));

	for(std::int64_t i = 0; i < size; ++i)
		{
			const float value                   = static_cast<float>(i % 1024) / 256.0f - 2.0f;
			values[static_cast<std::size_t>(i)] = positive ? std::abs(value) + 0.25f : value;
		}

	return values;
}

template <vext::Op Kp>
std::vector<float>
make_unary_values(
	const std::int64_t size)
{
	return make_values(size, Kp == vext::Op::LOG || Kp == vext::Op::SQRT || Kp == vext::Op::POW);
}

template <vext::Op Kp>
struct UnaryFunctor
{
	__host__ __device__ float
	operator()(
		const float value) const
	{
		if constexpr(Kp == vext::Op::ABS)
			{
				return fabsf(value);
			}
		else if constexpr(Kp == vext::Op::SIN)
			{
				return sinf(value);
			}
		else if constexpr(Kp == vext::Op::COS)
			{
				return cosf(value);
			}
		else if constexpr(Kp == vext::Op::TANH)
			{
				return tanhf(value);
			}
		else if constexpr(Kp == vext::Op::NEG)
			{
				return -value;
			}
		else if constexpr(Kp == vext::Op::EXP)
			{
				return expf(value);
			}
		else if constexpr(Kp == vext::Op::LOG)
			{
				return logf(value);
			}
		else if constexpr(Kp == vext::Op::SQRT)
			{
				return sqrtf(value);
			}
		else if constexpr(Kp == vext::Op::SQUARE)
			{
				return value * value;
			}
		else if constexpr(Kp == vext::Op::ROUND)
			{
				return roundf(value);
			}
		else if constexpr(Kp == vext::Op::SIGMOID)
			{
				return 1.0f / (1.0f + expf(-value));
			}
		else if constexpr(Kp == vext::Op::SOFT_RELU)
			{
				return logf(1.0f + expf(value));
			}
		else if constexpr(Kp == vext::Op::RELU)
			{
				return value > 0.0f ? value : 0.0f;
			}
		else if constexpr(Kp == vext::Op::LEAKY_RELU)
			{
				return value > 0.0f ? value : UNARY_PARAM_A * value;
			}
		else if constexpr(Kp == vext::Op::ELU)
			{
				return value > 0.0f ? value : UNARY_PARAM_A * (expf(value) - 1.0f);
			}
		else if constexpr(Kp == vext::Op::SWISH)
			{
				return value / (1.0f + expf(-UNARY_PARAM_A * value));
			}
		else if constexpr(Kp == vext::Op::LINEAR)
			{
				return UNARY_PARAM_A * value + UNARY_PARAM_B;
			}
		else if constexpr(Kp == vext::Op::CLIP)
			{
				return fmaxf(UNARY_PARAM_A, fminf(UNARY_PARAM_B, value));
			}
		else if constexpr(Kp == vext::Op::POW)
			{
				return UNARY_PARAM_A * powf(value, UNARY_PARAM_B);
			}
		else if constexpr(Kp == vext::Op::SOFTMIN)
			{
				return expf(-value);
			}
		else
			{
				return expf(value);
			}
	}
};

template <vext::Op Kp>
struct NormalizeFunctor
{
	float sum;

	__host__ __device__ float
	operator()(
		const float value) const
	{
		if constexpr(Kp == vext::Op::LOGSOFTMAX)
			{
				return logf(value / sum);
			}
		else
			{
				return value / sum;
			}
	}
};

template <vext::Op Kp>
void
run_vext_unary(
	vext::Tensor<float, vext::Backend::CUDA>& values)
{
	if constexpr(Kp == vext::Op::LEAKY_RELU || Kp == vext::Op::ELU || Kp == vext::Op::SWISH)
		{
			vext::unary<Kp>(values, UNARY_PARAM_A);
		}
	else if constexpr(Kp == vext::Op::LINEAR || Kp == vext::Op::CLIP || Kp == vext::Op::POW)
		{
			vext::unary<Kp>(values, UNARY_PARAM_A, UNARY_PARAM_B);
		}
	else
		{
			vext::unary<Kp>(values);
		}
}

template <vext::Op Kp>
void
run_thrust_unary(
	const DeviceBuffer<float>& input,
	DeviceBuffer<float>&       out,
	const std::int32_t         size)
{
	auto first  = thrust::device_pointer_cast(input.data());
	auto result = thrust::device_pointer_cast(out.data());
	thrust::transform(thrust::device, first, first + size, result, UnaryFunctor<Kp>{});

	if constexpr(Kp == vext::Op::SOFTMAX || Kp == vext::Op::SOFTMIN || Kp == vext::Op::LOGSOFTMAX)
		{
			const float sum = thrust::reduce(thrust::device, result, result + size, 0.0f, thrust::plus<float>{});
			thrust::transform(thrust::device, result, result + size, result, NormalizeFunctor<Kp>{ sum });
		}
}

template <vext::Op Kp>
void
BM_VextCudaUnary(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		return;

	const std::uint32_t size  = static_cast<std::uint32_t>(state.range(0));
	const auto          input = make_unary_values<Kp>(size);

	vext::Tensor<float, vext::Backend::CUDA> values(size);

	const CudaEventTimer timer;

	values.set_from(input);
	run_vext_unary<Kp>(values);

	check_cuda(cudaDeviceSynchronize(), "Warming up vext CUDA unary operation");

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			run_vext_unary<Kp>(values);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(values.data());
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <vext::Op Kp>
void
BM_ThrustCudaUnary(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::int32_t size = static_cast<std::int32_t>(state.range(0));

	DeviceBuffer<float> input(size);
	DeviceBuffer<float> out(size);

	const CudaEventTimer timer;

	input.fill_from_host(make_unary_values<Kp>(size));
	run_thrust_unary<Kp>(input, out, size);

	check_cuda(cudaDeviceSynchronize(), "Warming up Thrust CUDA unary operation");

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			run_thrust_unary<Kp>(input, out, size);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <vext::Op Kp>
struct BinaryFunctor
{
	__host__ __device__ float
	operator()(
		const float lhs,
		const float rhs) const
	{
		if constexpr(Kp == vext::Op::ADD)
			{
				return lhs + rhs;
			}
		else if constexpr(Kp == vext::Op::SUB)
			{
				return lhs - rhs;
			}
		else if constexpr(Kp == vext::Op::MUL)
			{
				return lhs * rhs;
			}
		else if constexpr(Kp == vext::Op::DIV)
			{
				return lhs / rhs;
			}
		else if constexpr(Kp == vext::Op::POW)
			{
				return powf(lhs, rhs);
			}
		else if constexpr(Kp == vext::Op::MIN)
			{
				return fminf(lhs, rhs);
			}
		else if constexpr(Kp == vext::Op::MAX)
			{
				return fmaxf(lhs, rhs);
			}
		else
			{
				return fmaxf(0.0f, lhs) + rhs * fminf(0.0f, lhs);
			}
	}
};

template <vext::Op Kp>
struct BroadcastFunctor
{
	const float* matrix;
	const float* row;

	std::int32_t cols;

	__host__ __device__ float
	operator()(
		const std::int32_t index) const
	{
		return BinaryFunctor<Kp>{}(matrix[index], row[index % cols]);
	}
};

template <vext::Op Kp>
void
BM_VextCudaBinary(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint32_t size = static_cast<std::uint32_t>(state.range(0));

	vext::Tensor<float, vext::Backend::CUDA> lhs(size);
	vext::Tensor<float, vext::Backend::CUDA> rhs(size);
	vext::Tensor<float, vext::Backend::CUDA> out(size);

	const CudaEventTimer timer;

	lhs.set_from(make_values(size, Kp == vext::Op::POW));
	rhs.set_from(make_values(size, true));
	vext::binary<Kp>(lhs, rhs, out);

	check_cuda(cudaDeviceSynchronize(), "Warming up vext CUDA binary operation");

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			vext::binary<Kp>(lhs, rhs, out);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <vext::Op Kp>
void
BM_ThrustCudaBinary(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::int32_t size = static_cast<std::int32_t>(state.range(0));

	DeviceBuffer<float> lhs(size);
	DeviceBuffer<float> rhs(size);
	DeviceBuffer<float> out(size);

	const CudaEventTimer timer;

	lhs.fill_from_host(make_values(size, Kp == vext::Op::POW));
	rhs.fill_from_host(make_values(size, true));
	thrust::transform(thrust::device, thrust::device_pointer_cast(lhs.data()), thrust::device_pointer_cast(lhs.data()) + size, thrust::device_pointer_cast(rhs.data()), thrust::device_pointer_cast(out.data()), BinaryFunctor<Kp>{});

	check_cuda(cudaDeviceSynchronize(), "Warming up Thrust CUDA binary operation");

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			thrust::transform(thrust::device, thrust::device_pointer_cast(lhs.data()), thrust::device_pointer_cast(lhs.data()) + size, thrust::device_pointer_cast(rhs.data()), thrust::device_pointer_cast(out.data()), BinaryFunctor<Kp>{});
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <vext::Op Kp>
void
BM_VextCudaBroadcast(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		return;

	const std::uint32_t rows = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t cols = static_cast<std::uint32_t>(state.range(1));

	vext::Tensor<float, vext::Backend::CUDA> matrix(rows, cols);
	vext::Tensor<float, vext::Backend::CUDA> row(cols);
	vext::Tensor<float, vext::Backend::CUDA> out(rows, cols);

	const CudaEventTimer timer;

	matrix.set_from(make_values(static_cast<std::int64_t>(rows) * cols, Kp == vext::Op::POW));
	row.set_from(make_values(cols, true));
	vext::binary<Kp>(matrix, row, out);

	check_cuda(cudaDeviceSynchronize(), "Warming up vext CUDA broadcast operation");

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			vext::binary<Kp>(matrix, row, out);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * rows * cols);
}

template <vext::Op Kp>
void
BM_ThrustCudaBroadcast(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::int32_t rows = static_cast<std::int32_t>(state.range(0));
	const std::int32_t cols = static_cast<std::int32_t>(state.range(1));
	const std::int32_t size = rows * cols;

	DeviceBuffer<float> matrix(size);
	DeviceBuffer<float> row(cols);
	DeviceBuffer<float> out(size);

	const CudaEventTimer timer;

	matrix.fill_from_host(make_values(size, Kp == vext::Op::POW));
	row.fill_from_host(make_values(cols, true));
	thrust::transform(thrust::device, thrust::make_counting_iterator<std::int32_t>(0), thrust::make_counting_iterator(size), thrust::device_pointer_cast(out.data()), BroadcastFunctor<Kp>{ matrix.data(), row.data(), cols });

	check_cuda(cudaDeviceSynchronize(), "Warming up Thrust CUDA broadcast operation");

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			thrust::transform(thrust::device, thrust::make_counting_iterator<std::int32_t>(0), thrust::make_counting_iterator(size), thrust::device_pointer_cast(out.data()), BroadcastFunctor<Kp>{ matrix.data(), row.data(), cols });
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <vext::Op Kp>
struct LogicalFunctor
{
	__host__ __device__
		std::uint8_t
		operator()(
			const float lhs,
			const float rhs) const
	{
		if constexpr(Kp == vext::Op::EQUAL)
			{
				return lhs == rhs;
			}
		else if constexpr(Kp == vext::Op::NOT_EQUAL)
			{
				return lhs != rhs;
			}
		else if constexpr(Kp == vext::Op::LESS)
			{
				return lhs < rhs;
			}
		else if constexpr(Kp == vext::Op::LESS_EQUAL)
			{
				return lhs <= rhs;
			}
		else if constexpr(Kp == vext::Op::GREATER)
			{
				return lhs > rhs;
			}
		else
			{
				return lhs >= rhs;
			}
	}
};

template <vext::Op Kp>
void
BM_VextCudaLogical(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint32_t size = static_cast<std::uint32_t>(state.range(0));

	vext::Tensor<float, vext::Backend::CUDA>        lhs(size);
	vext::Tensor<float, vext::Backend::CUDA>        rhs(size);
	vext::Tensor<std::uint8_t, vext::Backend::CUDA> out(size);

	const CudaEventTimer timer;

	lhs.set_from(make_values(size));
	rhs.set_from(make_values(size, true));
	vext::logical<Kp>(lhs, rhs, out);

	check_cuda(cudaDeviceSynchronize(), "Warming up vext CUDA logical operation");

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			vext::logical<Kp>(lhs, rhs, out);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <vext::Op Kp>
void
BM_ThrustCudaLogical(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::int32_t size = static_cast<std::int32_t>(state.range(0));

	DeviceBuffer<float>        lhs(size);
	DeviceBuffer<float>        rhs(size);
	DeviceBuffer<std::uint8_t> out(size);

	const CudaEventTimer timer;

	lhs.fill_from_host(make_values(size));
	rhs.fill_from_host(make_values(size, true));
	thrust::transform(thrust::device, thrust::device_pointer_cast(lhs.data()), thrust::device_pointer_cast(lhs.data()) + size, thrust::device_pointer_cast(rhs.data()), thrust::device_pointer_cast(out.data()), LogicalFunctor<Kp>{});

	check_cuda(cudaDeviceSynchronize(), "Warming up Thrust CUDA logical operation");

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			thrust::transform(thrust::device, thrust::device_pointer_cast(lhs.data()), thrust::device_pointer_cast(lhs.data()) + size, thrust::device_pointer_cast(rhs.data()), thrust::device_pointer_cast(out.data()), LogicalFunctor<Kp>{});
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * size);
}

struct SquareFunctor
{
	__host__ __device__ float
	operator()(const float value) const
	{
		return value * value;
	}
};

struct SquaredDifferenceFunctor
{
	float mean;

	__host__ __device__ float
	operator()(const float value) const
	{
		const float difference = value - mean;
		return difference * difference;
	}
};

template <vext::Op Kp>
float
thrust_reduce(
	const DeviceBuffer<float>& values,
	const std::int32_t         size)
{
	auto first = thrust::device_pointer_cast(values.data());

	if constexpr(Kp == vext::Op::SUM || Kp == vext::Op::MEAN)
		{
			const float sum = thrust::reduce(thrust::device, first, first + size, 0.0f, thrust::plus<float>{});
			return Kp == vext::Op::MEAN ? sum / size : sum;
		}
	else if constexpr(Kp == vext::Op::PROD)
		{
			return thrust::reduce(thrust::device, first, first + size, 1.0f, thrust::multiplies<float>{});
		}
	else if constexpr(Kp == vext::Op::MIN)
		{
			return thrust::reduce(thrust::device, first, first + size, std::numeric_limits<float>::max(), thrust::minimum<float>{});
		}
	else if constexpr(Kp == vext::Op::MAX)
		{
			return thrust::reduce(thrust::device, first, first + size, std::numeric_limits<float>::lowest(), thrust::maximum<float>{});
		}
	else if constexpr(Kp == vext::Op::L2_NORM)
		{
			return std::sqrt(thrust::transform_reduce(thrust::device, first, first + size, SquareFunctor{}, 0.0f, thrust::plus<float>{}));
		}
	else
		{
			const float mean     = thrust::reduce(thrust::device, first, first + size, 0.0f, thrust::plus<float>{}) / size;
			const float variance = thrust::transform_reduce(thrust::device, first, first + size, SquaredDifferenceFunctor{ mean }, 0.0f, thrust::plus<float>{}) / size;
			return Kp == vext::Op::STD ? std::sqrt(variance) : variance;
		}
}

template <vext::Op Kp>
void
BM_VextCudaReduction(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint32_t size = static_cast<std::uint32_t>(state.range(0));

	vext::Tensor<float, vext::Backend::CUDA> values(size);
	vext::Tensor<float, vext::Backend::CUDA> out(1);

	const CudaEventTimer timer;

	values.set_from(make_values(size, Kp == vext::Op::PROD));
	vext::reduction<Kp>(values, vext::core::no_value_t{}, out);

	check_cuda(cudaDeviceSynchronize(), "Warming up vext CUDA reduction");

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			vext::reduction<Kp>(values, vext::core::no_value_t{}, out);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * size);
}

template <vext::Op Kp>
void
BM_ThrustCudaReduction(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::int32_t size = static_cast<std::int32_t>(state.range(0));

	DeviceBuffer<float> values(size);

	const CudaEventTimer timer;

	values.fill_from_host(make_values(size, Kp == vext::Op::PROD));
	float out = 0.0f;
	out       = thrust_reduce<Kp>(values, size);

	check_cuda(cudaDeviceSynchronize(), "Warming up Thrust CUDA reduction");

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			out = thrust_reduce<Kp>(values, size);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out);
		}

	state.SetItemsProcessed(state.iterations() * size);
}

struct RowIndexFunctor
{
	std::int32_t cols;

	__host__ __device__
		std::int32_t
		operator()(
			const std::int32_t index) const
	{
		return index / cols;
	}
};

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

template <vext::Op Kp>
void
run_thrust_axis_reduction(
	const DeviceBuffer<float>& values,
	DeviceBuffer<float>&       means,
	DeviceBuffer<float>&       out,
	const std::int32_t         rows,
	const std::int32_t         cols)
{
	const std::int32_t size           = rows * cols;
	auto               indices        = thrust::make_counting_iterator<std::int32_t>(0);
	auto               keys           = thrust::make_transform_iterator(indices, RowIndexFunctor{ cols });
	auto               first          = thrust::device_pointer_cast(values.data());
	auto               result         = thrust::device_pointer_cast(out.data());
	auto               discarded_keys = thrust::make_discard_iterator();

	if constexpr(Kp == vext::Op::SUM || Kp == vext::Op::MEAN)
		{
			thrust::reduce_by_key(thrust::device, keys, keys + size, first, discarded_keys, result, thrust::equal_to<std::int32_t>{}, thrust::plus<float>{});

			if constexpr(Kp == vext::Op::MEAN)
				{
					thrust::transform(thrust::device, result, result + rows, result, ScaleFunctor<false>{ 1.0f / cols });
				}
		}
	else if constexpr(Kp == vext::Op::PROD)
		{
			thrust::reduce_by_key(thrust::device, keys, keys + size, first, discarded_keys, result, thrust::equal_to<std::int32_t>{}, thrust::multiplies<float>{});
		}
	else if constexpr(Kp == vext::Op::MIN)
		{
			thrust::reduce_by_key(thrust::device, keys, keys + size, first, discarded_keys, result, thrust::equal_to<std::int32_t>{}, thrust::minimum<float>{});
		}
	else if constexpr(Kp == vext::Op::MAX)
		{
			thrust::reduce_by_key(thrust::device, keys, keys + size, first, discarded_keys, result, thrust::equal_to<std::int32_t>{}, thrust::maximum<float>{});
		}
	else if constexpr(Kp == vext::Op::L2_NORM)
		{
			auto squares = thrust::make_transform_iterator(first, SquareFunctor{});
			thrust::reduce_by_key(thrust::device, keys, keys + size, squares, discarded_keys, result, thrust::equal_to<std::int32_t>{}, thrust::plus<float>{});
			thrust::transform(thrust::device, result, result + rows, result, ScaleFunctor<true>{ 1.0f });
		}
	else
		{
			auto mean_result = thrust::device_pointer_cast(means.data());
			thrust::reduce_by_key(thrust::device, keys, keys + size, first, discarded_keys, mean_result, thrust::equal_to<std::int32_t>{}, thrust::plus<float>{});
			thrust::transform(thrust::device, mean_result, mean_result + rows, mean_result, ScaleFunctor<false>{ 1.0f / cols });
			auto differences = thrust::make_transform_iterator(indices, RowSquaredDifferenceFunctor{ values.data(), means.data(), cols });
			thrust::reduce_by_key(thrust::device, keys, keys + size, differences, discarded_keys, result, thrust::equal_to<std::int32_t>{}, thrust::plus<float>{});

			if constexpr(Kp == vext::Op::STD)
				{
					thrust::transform(thrust::device, result, result + rows, result, ScaleFunctor<true>{ 1.0f / cols });
				}
			else
				{
					thrust::transform(thrust::device, result, result + rows, result, ScaleFunctor<false>{ 1.0f / cols });
				}
		}
}

template <vext::Op Kp>
void
BM_VextCudaAxisReduction(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint32_t rows = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t cols = static_cast<std::uint32_t>(state.range(1));

	vext::Tensor<float, vext::Backend::CUDA> values(rows, cols);
	vext::Tensor<float, vext::Backend::CUDA> out(rows);

	const CudaEventTimer timer;

	values.set_from(make_values(static_cast<std::int64_t>(rows) * cols, Kp == vext::Op::PROD));
	vext::reduction<Kp>(values, vext::axes({ 1 }), out);

	check_cuda(cudaDeviceSynchronize(), "Warming up vext CUDA axis reduction");

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			vext::reduction<Kp>(values, vext::axes({ 1 }), out);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * rows * cols);
}

template <vext::Op Kp>
void
BM_ThrustCudaAxisReduction(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		return;

	const std::int32_t rows = static_cast<std::int32_t>(state.range(0));
	const std::int32_t cols = static_cast<std::int32_t>(state.range(1));

	DeviceBuffer<float> values(static_cast<std::int64_t>(rows) * cols);
	DeviceBuffer<float> means(rows);
	DeviceBuffer<float> out(rows);

	const CudaEventTimer timer;

	values.fill_from_host(make_values(static_cast<std::int64_t>(rows) * cols, Kp == vext::Op::PROD));
	run_thrust_axis_reduction<Kp>(values, means, out, rows, cols);

	check_cuda(cudaDeviceSynchronize(), "Warming up Thrust CUDA axis reduction");

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			run_thrust_axis_reduction<Kp>(values, means, out, rows, cols);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * rows * cols);
}

void
BM_VextCudaMatmul(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		return;

	const std::uint32_t rows   = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t shared = static_cast<std::uint32_t>(state.range(1));
	const std::uint32_t cols   = static_cast<std::uint32_t>(state.range(2));
	const auto          zeros  = std::vector<float>(static_cast<std::size_t>(rows) * cols, 0.0f);

	vext::Tensor<float, vext::Backend::CUDA> lhs(rows, shared);
	vext::Tensor<float, vext::Backend::CUDA> rhs(shared, cols);
	vext::Tensor<float, vext::Backend::CUDA> out(rows, cols);

	const CudaEventTimer timer;

	lhs.set_from(make_values(static_cast<std::int64_t>(rows) * shared));
	rhs.set_from(make_values(static_cast<std::int64_t>(shared) * cols));
	vext::matmul(lhs, rhs, out);

	check_cuda(cudaDeviceSynchronize(), "Warming up vext CUDA matrix multiplication");

	for([[maybe_unused]] auto iteration : state)
		{
			state.PauseTiming();
			out.set_from(zeros);
			state.ResumeTiming();
			timer.start();
			vext::matmul(lhs, rhs, out);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.counters["FLOP/s"] = benchmark::Counter(static_cast<double>(state.iterations()) * 2.0 * rows * shared * cols, benchmark::Counter::kIsRate);
}

void
BM_CublasCudaMatmul(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::int32_t rows   = static_cast<std::int32_t>(state.range(0));
	const std::int32_t shared = static_cast<std::int32_t>(state.range(1));
	const std::int32_t cols   = static_cast<std::int32_t>(state.range(2));

	DeviceBuffer<float> lhs(static_cast<std::int64_t>(rows) * shared);
	DeviceBuffer<float> rhs(static_cast<std::int64_t>(shared) * cols);
	DeviceBuffer<float> out(static_cast<std::int64_t>(rows) * cols);

	const CublasHandle   handle;
	const CudaEventTimer timer;

	const float alpha = 1.0f;
	const float beta  = 0.0f;

	lhs.fill_from_host(make_values(static_cast<std::int64_t>(rows) * shared));
	rhs.fill_from_host(make_values(static_cast<std::int64_t>(shared) * cols));
	check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, shared, &alpha, rhs.data(), cols, lhs.data(), shared, &beta, out.data(), cols), "Warming up cuBLAS SGEMM");

	check_cuda(cudaDeviceSynchronize(), "Synchronizing cuBLAS SGEMM warmup");

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, shared, &alpha, rhs.data(), cols, lhs.data(), shared, &beta, out.data(), cols), "Running cuBLAS SGEMM");
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.counters["FLOP/s"] = benchmark::Counter(static_cast<double>(state.iterations()) * 2.0 * rows * shared * cols, benchmark::Counter::kIsRate);
}

struct CsrData
{
	std::vector<float>         values;
	std::vector<std::uint32_t> head;
	std::vector<std::uint32_t> tail;
};

CsrData
make_csr_data()
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

template <vext::Op Kp>
std::vector<float>
make_cusparse_values(
	const CsrData& data,
	const bool     use_values)
{
	std::vector<float> values(data.values.size(), 1.0f);

	if(use_values)
		{
			values = data.values;
		}

	if constexpr(Kp == vext::Op::MEAN)
		{
			for(float& value : values)
				{
					value /= static_cast<float>(CSR_DEGREE);
				}
		}

	return values;
}

template <vext::Op Kp>
void
BM_VextCudaCsrScatter(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		return;

	const CsrData data       = make_csr_data();
	const auto    src_values = make_values(CSR_ROWS * CSR_FEATURES);
	const auto    zeros      = std::vector<float>(CSR_ROWS * CSR_FEATURES, 0.0f);

	vext::Tensor<float, vext::Backend::CUDA>         src(CSR_ROWS, CSR_FEATURES);
	vext::Tensor<std::uint32_t, vext::Backend::CUDA> head(CSR_ROWS + 1);
	vext::Tensor<std::uint32_t, vext::Backend::CUDA> tail(CSR_ROWS * CSR_DEGREE);
	vext::Tensor<float, vext::Backend::CUDA>         out(CSR_ROWS, CSR_FEATURES);

	const CudaEventTimer timer;

	src.set_from(src_values);
	head.set_from(data.head);
	tail.set_from(data.tail);
	vext::csr_scatter<Kp>(src, head, tail, out);

	check_cuda(cudaDeviceSynchronize(), "Warming up vext CUDA CSR scatter");

	for([[maybe_unused]] auto iteration : state)
		{
			state.PauseTiming();
			out.set_from(zeros);
			state.ResumeTiming();
			timer.start();
			vext::csr_scatter<Kp>(src, head, tail, out);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * CSR_ROWS * CSR_DEGREE * CSR_FEATURES);
}

template <vext::Op Kp>
void
BM_CusparseCudaCsrScatter(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const CsrData data = make_csr_data();

	DeviceBuffer<std::uint32_t> head(CSR_ROWS + 1);
	DeviceBuffer<std::uint32_t> tail(CSR_ROWS * CSR_DEGREE);
	DeviceBuffer<float>         values(CSR_ROWS * CSR_DEGREE);
	DeviceBuffer<float>         src(CSR_ROWS * CSR_FEATURES);
	DeviceBuffer<float>         out(CSR_ROWS * CSR_FEATURES);

	const CusparseHandle handle;
	const CudaEventTimer timer;

	head.fill_from_host(data.head);
	tail.fill_from_host(data.tail);
	values.fill_from_host(make_cusparse_values<Kp>(data, false));
	src.fill_from_host(make_values(CSR_ROWS * CSR_FEATURES));

	cusparseSpMatDescr_t matrix    = nullptr;
	cusparseDnMatDescr_t dense_src = nullptr;
	cusparseDnMatDescr_t dense_out = nullptr;

	check_cusparse(cusparseCreateCsr(&matrix, CSR_ROWS, CSR_ROWS, CSR_ROWS * CSR_DEGREE, head.data(), tail.data(), values.data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), "Creating the cuSPARSE CSR scatter matrix");
	check_cusparse(cusparseCreateDnMat(&dense_src, CSR_ROWS, CSR_FEATURES, CSR_FEATURES, src.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW), "Creating the cuSPARSE scatter source matrix");
	check_cusparse(cusparseCreateDnMat(&dense_out, CSR_ROWS, CSR_FEATURES, CSR_FEATURES, out.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW), "Creating the cuSPARSE scatter output matrix");

	const float alpha = 1.0f;
	const float beta  = 0.0f;

	std::size_t workspace_size = 0;

	check_cusparse(cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix, dense_src, &beta, dense_out, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &workspace_size), "Querying cuSPARSE SpMM workspace size");

	DeviceBuffer<std::uint8_t> workspace(std::max<std::size_t>(workspace_size, 1));
	check_cusparse(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix, dense_src, &beta, dense_out, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, workspace.data()), "Warming up cuSPARSE CSR scatter through SpMM");

	check_cuda(cudaDeviceSynchronize(), "Synchronizing cuSPARSE SpMM warmup");

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			check_cusparse(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix, dense_src, &beta, dense_out, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, workspace.data()), "Running cuSPARSE CSR scatter through SpMM");
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	check_cusparse(cusparseDestroyDnMat(dense_out), "Destroying the cuSPARSE scatter output matrix");
	check_cusparse(cusparseDestroyDnMat(dense_src), "Destroying the cuSPARSE scatter source matrix");
	check_cusparse(cusparseDestroySpMat(matrix), "Destroying the cuSPARSE CSR scatter matrix");

	state.SetItemsProcessed(state.iterations() * CSR_ROWS * CSR_DEGREE * CSR_FEATURES);
}

template <vext::Op Kp>
void
BM_VextCudaCsrSpmv(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const CsrData data = make_csr_data();

	vext::Tensor<float, vext::Backend::CUDA>         values(CSR_ROWS * CSR_DEGREE);
	vext::Tensor<std::uint32_t, vext::Backend::CUDA> head(CSR_ROWS + 1);
	vext::Tensor<std::uint32_t, vext::Backend::CUDA> tail(CSR_ROWS * CSR_DEGREE);
	vext::Tensor<float, vext::Backend::CUDA>         x(CSR_ROWS);
	vext::Tensor<float, vext::Backend::CUDA>         out(CSR_ROWS);

	const CudaEventTimer timer;

	values.set_from(data.values);
	head.set_from(data.head);
	tail.set_from(data.tail);
	x.set_from(make_values(CSR_ROWS));
	vext::csr_spmv<Kp>(values, head, tail, x, out);

	check_cuda(cudaDeviceSynchronize(), "Warming up vext CUDA CSR SpMV");

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			vext::csr_spmv<Kp>(values, head, tail, x, out);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	state.SetItemsProcessed(state.iterations() * CSR_ROWS * CSR_DEGREE);
}

template <vext::Op Kp>
void
BM_CusparseCudaCsrSpmv(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const CsrData data = make_csr_data();

	DeviceBuffer<std::uint32_t> head(CSR_ROWS + 1);
	DeviceBuffer<std::uint32_t> tail(CSR_ROWS * CSR_DEGREE);
	DeviceBuffer<float>         values(CSR_ROWS * CSR_DEGREE);
	DeviceBuffer<float>         x(CSR_ROWS);
	DeviceBuffer<float>         out(CSR_ROWS);

	const CusparseHandle handle;
	const CudaEventTimer timer;

	head.fill_from_host(data.head);
	tail.fill_from_host(data.tail);
	values.fill_from_host(make_cusparse_values<Kp>(data, true));
	x.fill_from_host(make_values(CSR_ROWS));

	cusparseSpMatDescr_t matrix    = nullptr;
	cusparseDnVecDescr_t dense_x   = nullptr;
	cusparseDnVecDescr_t dense_out = nullptr;

	check_cusparse(cusparseCreateCsr(&matrix, CSR_ROWS, CSR_ROWS, CSR_ROWS * CSR_DEGREE, head.data(), tail.data(), values.data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), "Creating the cuSPARSE CSR matrix");
	check_cusparse(cusparseCreateDnVec(&dense_x, CSR_ROWS, x.data(), CUDA_R_32F), "Creating the cuSPARSE input vector");
	check_cusparse(cusparseCreateDnVec(&dense_out, CSR_ROWS, out.data(), CUDA_R_32F), "Creating the cuSPARSE output vector");

	const float alpha = 1.0f;
	const float beta  = 0.0f;

	std::size_t workspace_size = 0;
	check_cusparse(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix, dense_x, &beta, dense_out, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &workspace_size), "Querying cuSPARSE SpMV workspace size");

	DeviceBuffer<std::uint8_t> workspace(std::max<std::size_t>(workspace_size, 1));
	check_cusparse(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix, dense_x, &beta, dense_out, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, workspace.data()), "Warming up cuSPARSE SpMV");

	check_cuda(cudaDeviceSynchronize(), "Synchronizing cuSPARSE SpMV warmup");

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			check_cusparse(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix, dense_x, &beta, dense_out, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, workspace.data()), "Running cuSPARSE SpMV");
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out.data());
		}

	check_cusparse(cusparseDestroyDnVec(dense_out), "Destroying the cuSPARSE output vector");
	check_cusparse(cusparseDestroyDnVec(dense_x), "Destroying the cuSPARSE input vector");
	check_cusparse(cusparseDestroySpMat(matrix), "Destroying the cuSPARSE CSR matrix");

	state.SetItemsProcessed(state.iterations() * CSR_ROWS * CSR_DEGREE);
}

}

#define REGISTER_CUDA_PAIR(FAMILY, OP, SIZE)                                                                              \
	BENCHMARK_TEMPLATE(BM_VextCuda##FAMILY, vext::Op::OP)->Name("Vext/CUDA/" #FAMILY "_" #OP)->Arg(SIZE)->UseManualTime(); \
	BENCHMARK_TEMPLATE(BM_ThrustCuda##FAMILY, vext::Op::OP)->Name("Thrust/CUDA/" #FAMILY "_" #OP)->Arg(SIZE)->UseManualTime()

#define REGISTER_CUDA_MATRIX_PAIR(FAMILY, OP)                                                                                                      \
	BENCHMARK_TEMPLATE(BM_VextCuda##FAMILY, vext::Op::OP)->Name("Vext/CUDA/" #FAMILY "_" #OP)->Args({ MATRIX_ROWS, MATRIX_COLS })->UseManualTime(); \
	BENCHMARK_TEMPLATE(BM_ThrustCuda##FAMILY, vext::Op::OP)->Name("Thrust/CUDA/" #FAMILY "_" #OP)->Args({ MATRIX_ROWS, MATRIX_COLS })->UseManualTime()

REGISTER_CUDA_PAIR(Unary, ABS, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, SIN, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, COS, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, TANH, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, NEG, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, EXP, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, LOG, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, SQRT, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, SQUARE, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, ROUND, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, SIGMOID, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, SOFT_RELU, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, RELU, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, SOFTMAX, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, SOFTMIN, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, LOGSOFTMAX, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, LEAKY_RELU, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, ELU, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, SWISH, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, LINEAR, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, CLIP, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Unary, POW, VECTOR_SIZE);

REGISTER_CUDA_PAIR(Binary, ADD, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Binary, SUB, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Binary, MUL, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Binary, DIV, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Binary, POW, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Binary, MIN, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Binary, MAX, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Binary, PRELU, VECTOR_SIZE);

REGISTER_CUDA_MATRIX_PAIR(Broadcast, ADD);
REGISTER_CUDA_MATRIX_PAIR(Broadcast, SUB);
REGISTER_CUDA_MATRIX_PAIR(Broadcast, MUL);
REGISTER_CUDA_MATRIX_PAIR(Broadcast, DIV);
REGISTER_CUDA_MATRIX_PAIR(Broadcast, POW);
REGISTER_CUDA_MATRIX_PAIR(Broadcast, MIN);
REGISTER_CUDA_MATRIX_PAIR(Broadcast, MAX);
REGISTER_CUDA_MATRIX_PAIR(Broadcast, PRELU);

REGISTER_CUDA_PAIR(Logical, EQUAL, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Logical, NOT_EQUAL, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Logical, LESS, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Logical, LESS_EQUAL, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Logical, GREATER, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Logical, GREATER_EQUAL, VECTOR_SIZE);

REGISTER_CUDA_PAIR(Reduction, SUM, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Reduction, MEAN, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Reduction, MIN, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Reduction, MAX, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Reduction, PROD, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Reduction, STD, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Reduction, VAR, VECTOR_SIZE);
REGISTER_CUDA_PAIR(Reduction, L2_NORM, VECTOR_SIZE);

REGISTER_CUDA_MATRIX_PAIR(AxisReduction, SUM);
REGISTER_CUDA_MATRIX_PAIR(AxisReduction, MEAN);
REGISTER_CUDA_MATRIX_PAIR(AxisReduction, MIN);
REGISTER_CUDA_MATRIX_PAIR(AxisReduction, MAX);
REGISTER_CUDA_MATRIX_PAIR(AxisReduction, PROD);
REGISTER_CUDA_MATRIX_PAIR(AxisReduction, STD);
REGISTER_CUDA_MATRIX_PAIR(AxisReduction, VAR);
REGISTER_CUDA_MATRIX_PAIR(AxisReduction, L2_NORM);

BENCHMARK_TEMPLATE(BM_VextCudaCsrScatter, vext::Op::SUM)->Name("Vext/CUDA/CSRScatter_SUM")->Arg(CSR_ROWS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CusparseCudaCsrScatter, vext::Op::SUM)->Name("cuSPARSE/CUDA/CSRScatter_SUM")->Arg(CSR_ROWS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_VextCudaCsrScatter, vext::Op::MEAN)->Name("Vext/CUDA/CSRScatter_MEAN")->Arg(CSR_ROWS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CusparseCudaCsrScatter, vext::Op::MEAN)->Name("cuSPARSE/CUDA/CSRScatter_MEAN")->Arg(CSR_ROWS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_VextCudaCsrSpmv, vext::Op::SUM)->Name("Vext/CUDA/CSRSpMV_SUM")->Arg(CSR_ROWS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CusparseCudaCsrSpmv, vext::Op::SUM)->Name("cuSPARSE/CUDA/CSRSpMV_SUM")->Arg(CSR_ROWS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_VextCudaCsrSpmv, vext::Op::MEAN)->Name("Vext/CUDA/CSRSpMV_MEAN")->Arg(CSR_ROWS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CusparseCudaCsrSpmv, vext::Op::MEAN)->Name("cuSPARSE/CUDA/CSRSpMV_MEAN")->Arg(CSR_ROWS)->UseManualTime();

BENCHMARK(BM_VextCudaMatmul)->Name("Vext/CUDA/Matmul_Square")->Args({ 512, 512, 512 })->UseManualTime();
BENCHMARK(BM_CublasCudaMatmul)->Name("cuBLAS/CUDA/Matmul_Square")->Args({ 512, 512, 512 })->UseManualTime();
BENCHMARK(BM_VextCudaMatmul)->Name("Vext/CUDA/Matmul_Rectangular")->Args({ 256, 512, 128 })->UseManualTime();
BENCHMARK(BM_CublasCudaMatmul)->Name("cuBLAS/CUDA/Matmul_Rectangular")->Args({ 256, 512, 128 })->UseManualTime();

#undef REGISTER_CUDA_MATRIX_PAIR
#undef REGISTER_CUDA_PAIR

BENCHMARK_MAIN();
