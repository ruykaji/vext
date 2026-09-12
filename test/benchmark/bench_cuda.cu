#include <benchmark/benchmark.h>

#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include <vext/core/cuda/allocator.cuh>
#include <vext/core/cuda/ops/csr_scatter.cuh>
#include <vext/core/cuda/ops/csr_spmv.cuh>
#include <vext/core/cuda/ops/elementwise_binary.cuh>
#include <vext/core/cuda/ops/elementwise_logical.cuh>
#include <vext/core/cuda/ops/elementwise_unary.cuh>
#include <vext/core/cuda/ops/linear_algebra.cuh>
#include <vext/core/cuda/ops/memory.cuh>
#include <vext/core/cuda/ops/reduction.cuh>
#include <vext/nn/layer/linear.hpp>
#include <vext/ops.hpp>
#include <vext/tensor.hpp>

namespace
{

constexpr std::uint32_t ELEMENT_COUNT         = 1U << 24U;
constexpr std::uint32_t INPLACE_ELEMENT_COUNT = 1U << 20U;
constexpr std::uint32_t MATRIX_SIZE           = 512U;
constexpr std::uint32_t CSR_ROWS              = 1U << 20U;
constexpr std::uint32_t CSR_FEATURES          = 64U;
constexpr std::uint32_t CSR_DEGREE            = 32U;
constexpr std::int32_t  ELEMENT_ITERS         = 16;
constexpr std::int32_t  MATMUL_ITERS          = 16;

#if defined(__GNUC__) || defined(__clang__)
#define VEXT_BENCHMARK_NOINLINE __attribute__((noinline))
#else
#define VEXT_BENCHMARK_NOINLINE
#endif

bool
has_cuda_device()
{
	std::int32_t count = 0;
	cudaError_t  err   = cudaGetDeviceCount(&count);

	return err == cudaSuccess && count > 0;
}

bool
skip_without_cuda_device(
	benchmark::State& state)
{
	if(!has_cuda_device())
		{
			state.SkipWithError("No CUDA-capable device is available");
			return true;
		}

	return false;
}

VEXT_BENCHMARK_NOINLINE void
observe_device_buffer(
	std::uint8_t* ptr)
{
	benchmark::DoNotOptimize(ptr);
	benchmark::ClobberMemory();
}

class CudaEventTimer
{
public:
	CudaEventTimer()
	{
		cudaEventCreate(&__start);
		cudaEventCreate(&__stop);
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
		cudaEventRecord(__start, 0);
	}

	double
	stop_seconds() const
	{
		cudaEventRecord(__stop, 0);
		cudaEventSynchronize(__stop);

		float milliseconds = 0.0f;
		cudaEventElapsedTime(&milliseconds, __start, __stop);

		return static_cast<double>(milliseconds) / 1000.0;
	}

private:
	cudaEvent_t __start = nullptr;
	cudaEvent_t __stop  = nullptr;
};

std::vector<float>
make_lhs_values(
	const std::uint32_t size)
{
	std::vector<float> values(size, 1.25f);
	return values;
}

std::vector<float>
make_rhs_values(
	const std::uint32_t size)
{
	std::vector<float> values(size, 2.0f);
	return values;
}

template <typename Tp>
Tp*
copy_to_device(
	const std::vector<Tp>& host)
{
	Tp* device = vext::core::cuda::allocator::allocate<Tp>(host.size());
	vext::core::cuda::ops::memcpy<Tp, vext::Backend::CUDA, vext::Backend::CPU>(device, host.data(), static_cast<std::uint32_t>(host.size()));
	return device;
}

template <typename Tp>
VEXT_BENCHMARK_NOINLINE void
observe_tensor(
	const vext::Tensor<Tp, vext::Backend::CUDA>& tensor)
{
	benchmark::DoNotOptimize(tensor.length());
}

template <vext::Op Kp>
void
BM_CudaBinaryKernel(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint32_t size = static_cast<std::uint32_t>(state.range(0));

	float* lhs = copy_to_device(make_lhs_values(size));
	float* rhs = copy_to_device(make_rhs_values(size));
	float* out = vext::core::cuda::allocator::allocate<float>(size);

	const CudaEventTimer timer;

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			vext::core::cuda::ops::binary<Kp>(out, lhs, rhs, size);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out);
		}

	cudaDeviceSynchronize();

	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * sizeof(float) * 3));

	vext::core::cuda::allocator::deallocate(lhs);
	vext::core::cuda::allocator::deallocate(rhs);
	vext::core::cuda::allocator::deallocate(out);
	vext::core::cuda::allocator::free();
}

template <vext::Op Kp>
void
BM_CudaBinaryTensor(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	{
		const std::uint32_t rows = static_cast<std::uint32_t>(state.range(0));
		const std::uint32_t cols = static_cast<std::uint32_t>(state.range(1));
		const std::uint32_t size = rows * cols;

		vext::Tensor<float, vext::Backend::CUDA> lhs(rows, cols);
		vext::Tensor<float, vext::Backend::CUDA> rhs(rows, cols);

		lhs.set_from(make_lhs_values(size));
		rhs.set_from(make_rhs_values(size));

		const CudaEventTimer timer;

		for([[maybe_unused]] auto iteration : state)
			{
				timer.start();

				if constexpr(Kp == vext::Op::ADD)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::binary<vext::Op::ADD>(lhs, rhs);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::SUB)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::binary<vext::Op::SUB>(lhs, rhs);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::MUL)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::binary<vext::Op::MUL>(lhs, rhs);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::DIV)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::binary<vext::Op::DIV>(lhs, rhs);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::POW)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::binary<vext::Op::POW>(lhs, rhs);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::PRELU)
					{
						vext::Tensor<float, vext::Backend::CUDA> out(lhs);
						vext::binary<vext::Op::PRELU>(out, rhs, out);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
			}

		cudaDeviceSynchronize();

		state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
		state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * sizeof(float) * 3));
	}

	vext::core::cuda::allocator::free();
}

template <vext::Op Kp>
void
BM_CudaBinaryTensorInPlace(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	{
		const std::uint32_t rows = static_cast<std::uint32_t>(state.range(0));
		const std::uint32_t cols = static_cast<std::uint32_t>(state.range(1));
		const std::uint32_t size = rows * cols;

		vext::Tensor<float, vext::Backend::CUDA> source(rows, cols);
		vext::Tensor<float, vext::Backend::CUDA> rhs(rows, cols);

		source.set_from(make_lhs_values(size));
		rhs.set_from(make_rhs_values(size));

		vext::Tensor<float, vext::Backend::CUDA> out(source);

		const CudaEventTimer timer;

		for([[maybe_unused]] auto iteration : state)
			{
				timer.start();

				if constexpr(Kp == vext::Op::ADD)
					{
						vext::binary<vext::Op::ADD>(out, rhs, out);
					}
				else if constexpr(Kp == vext::Op::SUB)
					{
						vext::binary<vext::Op::SUB>(out, rhs, out);
					}
				else if constexpr(Kp == vext::Op::MUL)
					{
						vext::binary<vext::Op::MUL>(out, rhs, out);
					}
				else if constexpr(Kp == vext::Op::DIV)
					{
						vext::binary<vext::Op::DIV>(out, rhs, out);
					}
				else if constexpr(Kp == vext::Op::POW)
					{
						vext::binary<vext::Op::POW>(out, rhs, out);
					}

				state.SetIterationTime(timer.stop_seconds());
				observe_tensor(out);
			}

		cudaDeviceSynchronize();
		state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
		state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * sizeof(float) * 3));
	}

	vext::core::cuda::allocator::free();
}

template <vext::Op Kp>
void
BM_CudaLogicalKernel(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint32_t size = static_cast<std::uint32_t>(state.range(0));

	float*        lhs = copy_to_device(make_lhs_values(size));
	float*        rhs = copy_to_device(make_rhs_values(size));
	std::uint8_t* out = vext::core::cuda::allocator::allocate<std::uint8_t>(size);

	const CudaEventTimer timer;

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			vext::core::cuda::ops::logical<Kp>(out, lhs, rhs, size);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out);
		}

	cudaDeviceSynchronize();
	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * (sizeof(float) * 2 + sizeof(std::uint8_t))));

	vext::core::cuda::allocator::deallocate(lhs);
	vext::core::cuda::allocator::deallocate(rhs);
	vext::core::cuda::allocator::deallocate(out);
	vext::core::cuda::allocator::free();
}

template <vext::Op Kp>
void
BM_CudaLogicalTensor(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	{
		const std::uint32_t rows = static_cast<std::uint32_t>(state.range(0));
		const std::uint32_t cols = static_cast<std::uint32_t>(state.range(1));
		const std::uint32_t size = rows * cols;

		vext::Tensor<float, vext::Backend::CUDA> lhs(rows, cols);
		vext::Tensor<float, vext::Backend::CUDA> rhs(rows, cols);

		lhs.set_from(make_lhs_values(size));
		rhs.set_from(make_rhs_values(size));

		const CudaEventTimer timer;

		for([[maybe_unused]] auto iteration : state)
			{
				timer.start();

				if constexpr(Kp == vext::Op::EQUAL)
					{
						const vext::Tensor<std::uint8_t, vext::Backend::CUDA> out = vext::logical<vext::Op::EQUAL>(lhs, rhs);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::NOT_EQUAL)
					{
						const vext::Tensor<std::uint8_t, vext::Backend::CUDA> out = vext::logical<vext::Op::NOT_EQUAL>(lhs, rhs);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::LESS)
					{
						const vext::Tensor<std::uint8_t, vext::Backend::CUDA> out = vext::logical<vext::Op::LESS>(lhs, rhs);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::LESS_EQUAL)
					{
						const vext::Tensor<std::uint8_t, vext::Backend::CUDA> out = vext::logical<vext::Op::LESS_EQUAL>(lhs, rhs);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::GREATER)
					{
						const vext::Tensor<std::uint8_t, vext::Backend::CUDA> out = vext::logical<vext::Op::GREATER>(lhs, rhs);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::GREATER_EQUAL)
					{
						const vext::Tensor<std::uint8_t, vext::Backend::CUDA> out = vext::logical<vext::Op::GREATER_EQUAL>(lhs, rhs);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
			}

		cudaDeviceSynchronize();

		state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
		state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * (sizeof(float) * 2 + sizeof(std::uint8_t))));
	}

	vext::core::cuda::allocator::free();
}

template <vext::Op Kp>
void
BM_CudaUnaryKernel(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint32_t size = static_cast<std::uint32_t>(state.range(0));

	float* values = copy_to_device(std::vector<float>(size, 0.5f));

	const CudaEventTimer timer;

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();

			if constexpr(Kp == vext::Op::LEAKY_RELU || Kp == vext::Op::ELU || Kp == vext::Op::SWISH)
				{
					vext::core::cuda::ops::unary<Kp>(values, size, 0.25f);
				}
			else if constexpr(Kp == vext::Op::LINEAR || Kp == vext::Op::CLIP || Kp == vext::Op::POW)
				{
					vext::core::cuda::ops::unary<Kp>(values, size, 0.75f, 1.25f);
				}
			else
				{
					vext::core::cuda::ops::unary<Kp>(values, size);
				}

			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(values);
		}

	cudaDeviceSynchronize();

	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * sizeof(float)));

	vext::core::cuda::allocator::deallocate(values);
	vext::core::cuda::allocator::free();
}

template <vext::Op Kp>
void
BM_CudaUnaryTensor(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	{
		const std::uint32_t rows = static_cast<std::uint32_t>(state.range(0));
		const std::uint32_t cols = static_cast<std::uint32_t>(state.range(1));
		const std::uint32_t size = rows * cols;

		vext::Tensor<float, vext::Backend::CUDA> source(rows, cols);
		source.set_from(std::vector<float>(size, 0.5f));

		const CudaEventTimer timer;

		for([[maybe_unused]] auto iteration : state)
			{
				vext::Tensor<float, vext::Backend::CUDA> out(source);
				timer.start();

				if constexpr(Kp == vext::Op::ABS)
					{
						vext::unary<vext::Op::ABS>(out);
					}
				else if constexpr(Kp == vext::Op::SIN)
					{
						vext::unary<vext::Op::SIN>(out);
					}
				else if constexpr(Kp == vext::Op::COS)
					{
						vext::unary<vext::Op::COS>(out);
					}
				else if constexpr(Kp == vext::Op::NEG)
					{
						vext::unary<vext::Op::NEG>(out);
					}
				else if constexpr(Kp == vext::Op::EXP)
					{
						vext::unary<vext::Op::EXP>(out);
					}
				else if constexpr(Kp == vext::Op::LOG)
					{
						vext::unary<vext::Op::LOG>(out);
					}
				else if constexpr(Kp == vext::Op::SQRT)
					{
						vext::unary<vext::Op::SQRT>(out);
					}
				else if constexpr(Kp == vext::Op::SQUARE)
					{
						vext::unary<vext::Op::SQUARE>(out);
					}
				else if constexpr(Kp == vext::Op::ROUND)
					{
						vext::unary<vext::Op::ROUND>(out);
					}
				else if constexpr(Kp == vext::Op::SIGMOID)
					{
						vext::unary<vext::Op::SIGMOID>(out);
					}
				else if constexpr(Kp == vext::Op::SOFT_RELU)
					{
						vext::unary<vext::Op::SOFT_RELU>(out);
					}
				else if constexpr(Kp == vext::Op::RELU)
					{
						vext::unary<vext::Op::RELU>(out);
					}
				else if constexpr(Kp == vext::Op::SOFTMAX)
					{
						vext::unary<vext::Op::SOFTMAX>(out);
					}
				else if constexpr(Kp == vext::Op::SOFTMIN)
					{
						vext::unary<vext::Op::SOFTMIN>(out);
					}
				else if constexpr(Kp == vext::Op::LOGSOFTMAX)
					{
						vext::unary<vext::Op::LOGSOFTMAX>(out);
					}
				else if constexpr(Kp == vext::Op::LEAKY_RELU)
					{
						vext::unary<vext::Op::LEAKY_RELU>(out, 0.25f);
					}
				else if constexpr(Kp == vext::Op::ELU)
					{
						vext::unary<vext::Op::ELU>(out, 0.25f);
					}
				else if constexpr(Kp == vext::Op::SWISH)
					{
						vext::unary<vext::Op::SWISH>(out, 0.25f);
					}
				else if constexpr(Kp == vext::Op::LINEAR)
					{
						vext::unary<vext::Op::LINEAR>(out, 0.75f, 1.25f);
					}
				else if constexpr(Kp == vext::Op::CLIP)
					{
						vext::unary<vext::Op::CLIP>(out, 0.25f, 0.75f);
					}
				else if constexpr(Kp == vext::Op::POW)
					{
						vext::unary<vext::Op::POW>(out, 0.75f, 1.25f);
					}

				state.SetIterationTime(timer.stop_seconds());
				observe_tensor(out);
			}

		cudaDeviceSynchronize();

		state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
		state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * sizeof(float)));
	}

	vext::core::cuda::allocator::free();
}

template <vext::Op Kp>
void
BM_CudaReductionKernel(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint32_t size = static_cast<std::uint32_t>(state.range(0));

	float* values = copy_to_device(std::vector<float>(size, 1.0f));
	float* out    = vext::core::cuda::allocator::allocate<float>(1);

	const std::vector<std::uint32_t> keep_dims{ 1 };
	const std::vector<std::uint32_t> keep_strides{ 0 };
	const std::vector<std::uint32_t> reduce_dims{ size };
	const std::vector<std::uint32_t> reduce_strides{ 1 };

	const CudaEventTimer timer;

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			vext::core::cuda::ops::reduce<Kp>(out, values, 1, size, keep_dims, keep_strides, reduce_dims, reduce_strides);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out);
		}

	cudaDeviceSynchronize();

	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * sizeof(float)));

	vext::core::cuda::allocator::deallocate(values);
	vext::core::cuda::allocator::deallocate(out);
	vext::core::cuda::allocator::free();
}

template <vext::Op Kp>
void
BM_CudaReductionTensor(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	{
		const std::uint32_t size = static_cast<std::uint32_t>(state.range(0));

		vext::Tensor<float, vext::Backend::CUDA> tensor(size);
		tensor.set_from(std::vector<float>(size, 1.0f));

		const CudaEventTimer timer;

		for([[maybe_unused]] auto iteration : state)
			{
				timer.start();

				if constexpr(Kp == vext::Op::SUM)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::reduction<vext::Op::SUM>(tensor);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::MEAN)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::reduction<vext::Op::MEAN>(tensor);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::MAX)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::reduction<vext::Op::MAX>(tensor);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::MIN)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::reduction<vext::Op::MIN>(tensor);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::PROD)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::reduction<vext::Op::PROD>(tensor);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::STD)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::reduction<vext::Op::STD>(tensor);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::VAR)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::reduction<vext::Op::VAR>(tensor);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
			}

		cudaDeviceSynchronize();

		state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
		state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * size * sizeof(float)));
	}

	vext::core::cuda::allocator::free();
}

std::vector<std::uint32_t>
make_csr_head(
	const std::uint32_t rows,
	const std::uint32_t degree)
{
	std::vector<std::uint32_t> head(rows + 1U, 0U);

	for(std::uint32_t row = 0; row <= rows; ++row)
		{
			head[row] = row * degree;
		}

	return head;
}

std::vector<std::uint32_t>
make_csr_tail(
	const std::uint32_t rows,
	const std::uint32_t degree)
{
	std::vector<std::uint32_t> tail(rows * degree, 0U);

	for(std::uint32_t row = 0; row < rows; ++row)
		{
			for(std::uint32_t edge = 0; edge < degree; ++edge)
				{
					tail[row * degree + edge] = (row + edge) % rows;
				}
		}

	return tail;
}

template <vext::Op Kp>
std::vector<float>
make_csr_output_seed(
	const std::uint32_t size)
{
	float value = 0.0f;

	if constexpr(Kp == vext::Op::MIN)
		{
			value = std::numeric_limits<float>::max();
		}
	else if constexpr(Kp == vext::Op::MAX)
		{
			value = std::numeric_limits<float>::lowest();
		}
	else if constexpr(Kp == vext::Op::PROD)
		{
			value = 1.0f;
		}

	std::vector<float> seed(size, value);
	return seed;
}

template <vext::Op Kp>
void
BM_CudaCsrScatterKernel(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint32_t rows     = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t features = static_cast<std::uint32_t>(state.range(1));
	const std::uint32_t degree   = static_cast<std::uint32_t>(state.range(2));
	const std::uint32_t size     = rows * features;

	float*         src      = copy_to_device(std::vector<float>(size, 1.25f));
	float*         out      = vext::core::cuda::allocator::allocate<float>(size);
	float*         out_seed = copy_to_device(make_csr_output_seed<Kp>(size));
	std::uint32_t* head     = copy_to_device(make_csr_head(rows, degree));
	std::uint32_t* tail     = copy_to_device(make_csr_tail(rows, degree));

	const CudaEventTimer timer;

	for([[maybe_unused]] auto iteration : state)
		{
			cudaMemcpy(out, out_seed, size * sizeof(float), cudaMemcpyDeviceToDevice);
			timer.start();
			vext::core::cuda::ops::csr_scatter<Kp>(out, src, head, tail, rows, features);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out);
		}

	cudaDeviceSynchronize();

	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * rows * degree * features));
	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * rows * degree * features * sizeof(float)));

	vext::core::cuda::allocator::deallocate(src);
	vext::core::cuda::allocator::deallocate(out);
	vext::core::cuda::allocator::deallocate(out_seed);
	vext::core::cuda::allocator::deallocate(head);
	vext::core::cuda::allocator::deallocate(tail);
	vext::core::cuda::allocator::free();
}

template <vext::Op Kp>
void
BM_CudaCsrScatterTensor(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	{
		const std::uint32_t rows     = static_cast<std::uint32_t>(state.range(0));
		const std::uint32_t features = static_cast<std::uint32_t>(state.range(1));
		const std::uint32_t degree   = static_cast<std::uint32_t>(state.range(2));
		const std::uint32_t size     = rows * features;

		vext::Tensor<float, vext::Backend::CUDA>         src(rows, features);
		vext::Tensor<std::uint32_t, vext::Backend::CUDA> head(rows + 1U);
		vext::Tensor<std::uint32_t, vext::Backend::CUDA> tail(rows * degree);

		src.set_from(std::vector<float>(size, 1.25f));
		head.set_from(make_csr_head(rows, degree));
		tail.set_from(make_csr_tail(rows, degree));

		const CudaEventTimer timer;

		for([[maybe_unused]] auto iteration : state)
			{
				timer.start();

				if constexpr(Kp == vext::Op::SUM)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::csr_scatter<vext::Op::SUM>(src, head, tail);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::MEAN)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::csr_scatter<vext::Op::MEAN>(src, head, tail);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::MAX)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::csr_scatter<vext::Op::MAX>(src, head, tail);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::MIN)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::csr_scatter<vext::Op::MIN>(src, head, tail);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::PROD)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::csr_scatter<vext::Op::PROD>(src, head, tail);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::STD)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::csr_scatter<vext::Op::STD>(src, head, tail);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::VAR)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::csr_scatter<vext::Op::VAR>(src, head, tail);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
			}

		cudaDeviceSynchronize();

		state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * rows * degree * features));
		state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * rows * degree * features * sizeof(float)));
	}

	vext::core::cuda::allocator::free();
}

template <vext::Op Kp>
void
BM_CudaCsrSpmvKernel(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint32_t rows   = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t degree = static_cast<std::uint32_t>(state.range(1));
	const std::uint32_t nnz    = rows * degree;

	float*         values = copy_to_device(std::vector<float>(nnz, 1.25f));
	float*         x      = copy_to_device(std::vector<float>(rows, 2.0f));
	float*         out    = vext::core::cuda::allocator::allocate<float>(rows);
	std::uint32_t* head   = copy_to_device(make_csr_head(rows, degree));
	std::uint32_t* tail   = copy_to_device(make_csr_tail(rows, degree));

	const CudaEventTimer timer;

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			vext::core::cuda::ops::csr_spmv<Kp>(out, values, head, tail, x, rows);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out);
		}

	cudaDeviceSynchronize();

	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * nnz));
	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * nnz * (sizeof(float) * 2 + sizeof(std::uint32_t))));

	vext::core::cuda::allocator::deallocate(values);
	vext::core::cuda::allocator::deallocate(x);
	vext::core::cuda::allocator::deallocate(out);
	vext::core::cuda::allocator::deallocate(head);
	vext::core::cuda::allocator::deallocate(tail);
	vext::core::cuda::allocator::free();
}

template <vext::Op Kp>
void
BM_CudaCsrSpmvTensor(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	{
		const std::uint32_t rows   = static_cast<std::uint32_t>(state.range(0));
		const std::uint32_t degree = static_cast<std::uint32_t>(state.range(1));
		const std::uint32_t nnz    = rows * degree;

		vext::Tensor<float, vext::Backend::CUDA>         values(nnz);
		vext::Tensor<std::uint32_t, vext::Backend::CUDA> head(rows + 1U);
		vext::Tensor<std::uint32_t, vext::Backend::CUDA> tail(nnz);
		vext::Tensor<float, vext::Backend::CUDA>         x(rows);

		values.set_from(std::vector<float>(nnz, 1.25f));
		head.set_from(make_csr_head(rows, degree));
		tail.set_from(make_csr_tail(rows, degree));
		x.set_from(std::vector<float>(rows, 2.0f));

		const CudaEventTimer timer;

		for([[maybe_unused]] auto iteration : state)
			{
				timer.start();

				if constexpr(Kp == vext::Op::SUM)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::csr_spmv<vext::Op::SUM>(values, head, tail, x);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::MEAN)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::csr_spmv<vext::Op::MEAN>(values, head, tail, x);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::MAX)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::csr_spmv<vext::Op::MAX>(values, head, tail, x);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::MIN)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::csr_spmv<vext::Op::MIN>(values, head, tail, x);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::PROD)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::csr_spmv<vext::Op::PROD>(values, head, tail, x);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::STD)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::csr_spmv<vext::Op::STD>(values, head, tail, x);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::Op::VAR)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = vext::csr_spmv<vext::Op::VAR>(values, head, tail, x);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
			}

		cudaDeviceSynchronize();

		state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * nnz));
		state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * nnz * (sizeof(float) * 2 + sizeof(std::uint32_t))));
	}

	vext::core::cuda::allocator::free();
}

void
BM_CudaMatmulKernel(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint32_t m = static_cast<std::uint32_t>(state.range(0));
	const std::uint32_t p = static_cast<std::uint32_t>(state.range(1));
	const std::uint32_t n = static_cast<std::uint32_t>(state.range(2));

	float* lhs = copy_to_device(std::vector<float>(m * p, 0.5f));
	float* rhs = copy_to_device(std::vector<float>(p * n, 0.25f));
	float* out = vext::core::cuda::allocator::allocate<float>(m * n);

	const CudaEventTimer timer;

	for([[maybe_unused]] auto iteration : state)
		{
			timer.start();
			vext::core::cuda::ops::matmul(out, lhs, rhs, m, p, n);
			state.SetIterationTime(timer.stop_seconds());
			benchmark::DoNotOptimize(out);
		}

	cudaDeviceSynchronize();

	state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * m * n));
	state.counters["flop"] = benchmark::Counter(static_cast<double>(state.iterations()) * 2.0 * m * n * p, benchmark::Counter::kIsRate);

	vext::core::cuda::allocator::deallocate(lhs);
	vext::core::cuda::allocator::deallocate(rhs);
	vext::core::cuda::allocator::deallocate(out);
	vext::core::cuda::allocator::free();
}

void
BM_CudaTensorMatmul(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	{
		const std::uint32_t m = static_cast<std::uint32_t>(state.range(0));
		const std::uint32_t p = static_cast<std::uint32_t>(state.range(1));
		const std::uint32_t n = static_cast<std::uint32_t>(state.range(2));

		const vext::Tensor<float, vext::Backend::CUDA> lhs(m, p);
		const vext::Tensor<float, vext::Backend::CUDA> rhs(p, n);
		const CudaEventTimer                           timer;

		for([[maybe_unused]] auto iteration : state)
			{
				timer.start();
				const vext::Tensor<float, vext::Backend::CUDA> out = vext::matmul(lhs, rhs);
				state.SetIterationTime(timer.stop_seconds());
				observe_tensor(out);
			}

		cudaDeviceSynchronize();

		state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * m * n));
		state.counters["flop"] = benchmark::Counter(static_cast<double>(state.iterations()) * 2.0 * m * n * p, benchmark::Counter::kIsRate);
	}

	vext::core::cuda::allocator::free();
}

void
BM_CudaNnLinearForward(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	{
		const std::uint32_t                                size = static_cast<std::uint32_t>(state.range(0));
		const vext::Tensor<float, vext::Backend::CUDA>     input(size, size);
		const vext::nn::layer::Linear<vext::Backend::CUDA> layer(size, size);

		const CudaEventTimer timer;

		for([[maybe_unused]] auto iteration : state)
			{
				timer.start();
				const vext::Tensor<float, vext::Backend::CUDA> out = layer(input);
				state.SetIterationTime(timer.stop_seconds());
				observe_tensor(out);
			}

		cudaDeviceSynchronize();

		state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size * size));
		state.counters["flop"] = benchmark::Counter(static_cast<double>(state.iterations()) * 2.0 * size * size * size, benchmark::Counter::kIsRate);
	}

	vext::core::cuda::allocator::free();
}

void
BM_CudaAllocatorSmall(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint64_t bytes = static_cast<std::uint64_t>(state.range(0));

	for([[maybe_unused]] auto iteration : state)
		{
			std::uint8_t* ptr = vext::core::cuda::allocator::allocate<std::uint8_t>(bytes);
			observe_device_buffer(ptr);
			vext::core::cuda::allocator::deallocate(ptr);
		}

	cudaDeviceSynchronize();

	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * bytes));

	vext::core::cuda::allocator::free();
}

void
BM_CudaAllocatorLarge(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	const std::uint64_t bytes = static_cast<std::uint64_t>(state.range(0));

	for([[maybe_unused]] auto iteration : state)
		{
			std::uint8_t* ptr = vext::core::cuda::allocator::allocate<std::uint8_t>(bytes);
			observe_device_buffer(ptr);
			vext::core::cuda::allocator::deallocate(ptr);
		}

	cudaDeviceSynchronize();

	state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * bytes));

	vext::core::cuda::allocator::free();
}

void
BM_CudaTensorConstruct(
	benchmark::State& state)
{
	if(skip_without_cuda_device(state))
		{
			return;
		}

	{
		const std::uint32_t rows = static_cast<std::uint32_t>(state.range(0));
		const std::uint32_t cols = static_cast<std::uint32_t>(state.range(1));

		for([[maybe_unused]] auto iteration : state)
			{
				const vext::Tensor<float, vext::Backend::CUDA> tensor(rows, cols);
				observe_tensor(tensor);
			}

		cudaDeviceSynchronize();

		state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * rows * cols));
	}

	vext::core::cuda::allocator::free();
}

}

BENCHMARK(BM_CudaAllocatorSmall)->Arg(4096)->Iterations(200000);
BENCHMARK(BM_CudaAllocatorLarge)->Arg(32 * 1024 * 1024)->Iterations(128);
BENCHMARK(BM_CudaTensorConstruct)->Args({ 4096, 4096 })->Iterations(32);

BENCHMARK_TEMPLATE(BM_CudaBinaryKernel, vext::Op::ADD)->Name("BM_CudaBinaryKernel/ADD")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensor, vext::Op::ADD)->Name("BM_CudaBinaryTensor/ADD")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensorInPlace, vext::Op::ADD)->Name("BM_CudaBinaryTensorInPlace/ADD")->Args({ INPLACE_ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryKernel, vext::Op::SUB)->Name("BM_CudaBinaryKernel/SUB")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensor, vext::Op::SUB)->Name("BM_CudaBinaryTensor/SUB")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensorInPlace, vext::Op::SUB)->Name("BM_CudaBinaryTensorInPlace/SUB")->Args({ INPLACE_ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryKernel, vext::Op::MUL)->Name("BM_CudaBinaryKernel/MUL")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensor, vext::Op::MUL)->Name("BM_CudaBinaryTensor/MUL")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensorInPlace, vext::Op::MUL)->Name("BM_CudaBinaryTensorInPlace/MUL")->Args({ INPLACE_ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryKernel, vext::Op::DIV)->Name("BM_CudaBinaryKernel/DIV")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensor, vext::Op::DIV)->Name("BM_CudaBinaryTensor/DIV")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensorInPlace, vext::Op::DIV)->Name("BM_CudaBinaryTensorInPlace/DIV")->Args({ INPLACE_ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryKernel, vext::Op::POW)->Name("BM_CudaBinaryKernel/POW")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensor, vext::Op::POW)->Name("BM_CudaBinaryTensor/POW")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensorInPlace, vext::Op::POW)->Name("BM_CudaBinaryTensorInPlace/POW")->Args({ INPLACE_ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryKernel, vext::Op::PRELU)->Name("BM_CudaBinaryKernel/PRELU")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensor, vext::Op::PRELU)->Name("BM_CudaBinaryTensor/PRELU")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryKernel, vext::Op::MIN)->Name("BM_CudaBinaryKernel/MIN/kernel_only")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryKernel, vext::Op::MAX)->Name("BM_CudaBinaryKernel/MAX/kernel_only")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();

BENCHMARK_TEMPLATE(BM_CudaLogicalKernel, vext::Op::EQUAL)->Name("BM_CudaLogicalKernel/EQUAL")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalTensor, vext::Op::EQUAL)->Name("BM_CudaLogicalTensor/EQUAL")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalKernel, vext::Op::NOT_EQUAL)->Name("BM_CudaLogicalKernel/NOT_EQUAL")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalTensor, vext::Op::NOT_EQUAL)->Name("BM_CudaLogicalTensor/NOT_EQUAL")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalKernel, vext::Op::LESS)->Name("BM_CudaLogicalKernel/LESS")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalTensor, vext::Op::LESS)->Name("BM_CudaLogicalTensor/LESS")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalKernel, vext::Op::LESS_EQUAL)->Name("BM_CudaLogicalKernel/LESS_EQUAL")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalTensor, vext::Op::LESS_EQUAL)->Name("BM_CudaLogicalTensor/LESS_EQUAL")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalKernel, vext::Op::GREATER)->Name("BM_CudaLogicalKernel/GREATER")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalTensor, vext::Op::GREATER)->Name("BM_CudaLogicalTensor/GREATER")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalKernel, vext::Op::GREATER_EQUAL)->Name("BM_CudaLogicalKernel/GREATER_EQUAL")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalTensor, vext::Op::GREATER_EQUAL)->Name("BM_CudaLogicalTensor/GREATER_EQUAL")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();

BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::ABS)->Name("BM_CudaUnaryKernel/ABS")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::ABS)->Name("BM_CudaUnaryTensor/ABS")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::SIN)->Name("BM_CudaUnaryKernel/SIN")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::SIN)->Name("BM_CudaUnaryTensor/SIN")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::COS)->Name("BM_CudaUnaryKernel/COS")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::COS)->Name("BM_CudaUnaryTensor/COS")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::TANH)->Name("BM_CudaUnaryKernel/TANH/kernel_only")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::NEG)->Name("BM_CudaUnaryKernel/NEG")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::NEG)->Name("BM_CudaUnaryTensor/NEG")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::EXP)->Name("BM_CudaUnaryKernel/EXP")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::EXP)->Name("BM_CudaUnaryTensor/EXP")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::LOG)->Name("BM_CudaUnaryKernel/LOG")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::LOG)->Name("BM_CudaUnaryTensor/LOG")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::SQRT)->Name("BM_CudaUnaryKernel/SQRT")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::SQRT)->Name("BM_CudaUnaryTensor/SQRT")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::SQUARE)->Name("BM_CudaUnaryKernel/SQUARE")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::SQUARE)->Name("BM_CudaUnaryTensor/SQUARE")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::ROUND)->Name("BM_CudaUnaryKernel/ROUND")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::ROUND)->Name("BM_CudaUnaryTensor/ROUND")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::SIGMOID)->Name("BM_CudaUnaryKernel/SIGMOID")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::SIGMOID)->Name("BM_CudaUnaryTensor/SIGMOID")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::SOFT_RELU)->Name("BM_CudaUnaryKernel/SOFT_RELU")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::SOFT_RELU)->Name("BM_CudaUnaryTensor/SOFT_RELU")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::RELU)->Name("BM_CudaUnaryKernel/RELU")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::RELU)->Name("BM_CudaUnaryTensor/RELU")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::SOFTMAX)->Name("BM_CudaUnaryKernel/SOFTMAX")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::SOFTMAX)->Name("BM_CudaUnaryTensor/SOFTMAX")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::SOFTMIN)->Name("BM_CudaUnaryKernel/SOFTMIN")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::SOFTMIN)->Name("BM_CudaUnaryTensor/SOFTMIN")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::LOGSOFTMAX)->Name("BM_CudaUnaryKernel/LOGSOFTMAX")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::LOGSOFTMAX)->Name("BM_CudaUnaryTensor/LOGSOFTMAX")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::LEAKY_RELU)->Name("BM_CudaUnaryKernel/LEAKY_RELU")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::LEAKY_RELU)->Name("BM_CudaUnaryTensor/LEAKY_RELU")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::ELU)->Name("BM_CudaUnaryKernel/ELU")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::ELU)->Name("BM_CudaUnaryTensor/ELU")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::SWISH)->Name("BM_CudaUnaryKernel/SWISH")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::SWISH)->Name("BM_CudaUnaryTensor/SWISH")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::LINEAR)->Name("BM_CudaUnaryKernel/LINEAR")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::LINEAR)->Name("BM_CudaUnaryTensor/LINEAR")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::CLIP)->Name("BM_CudaUnaryKernel/CLIP")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::CLIP)->Name("BM_CudaUnaryTensor/CLIP")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::Op::POW)->Name("BM_CudaUnaryKernel/UNARY_POW")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::Op::POW)->Name("BM_CudaUnaryTensor/UNARY_POW")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();

BENCHMARK_TEMPLATE(BM_CudaReductionKernel, vext::Op::SUM)->Name("BM_CudaReductionKernel/SUM")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionTensor, vext::Op::SUM)->Name("BM_CudaReductionTensor/SUM")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionKernel, vext::Op::MEAN)->Name("BM_CudaReductionKernel/MEAN")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionTensor, vext::Op::MEAN)->Name("BM_CudaReductionTensor/MEAN")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionKernel, vext::Op::MAX)->Name("BM_CudaReductionKernel/MAX")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionTensor, vext::Op::MAX)->Name("BM_CudaReductionTensor/MAX")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionKernel, vext::Op::MIN)->Name("BM_CudaReductionKernel/MIN")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionTensor, vext::Op::MIN)->Name("BM_CudaReductionTensor/MIN")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionKernel, vext::Op::PROD)->Name("BM_CudaReductionKernel/PROD")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionTensor, vext::Op::PROD)->Name("BM_CudaReductionTensor/PROD")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionKernel, vext::Op::STD)->Name("BM_CudaReductionKernel/STD")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionTensor, vext::Op::STD)->Name("BM_CudaReductionTensor/STD")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionKernel, vext::Op::VAR)->Name("BM_CudaReductionKernel/VAR")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionTensor, vext::Op::VAR)->Name("BM_CudaReductionTensor/VAR")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();

BENCHMARK_TEMPLATE(BM_CudaCsrScatterKernel, vext::Op::SUM)->Name("BM_CudaCsrScatterKernel/SUM")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterTensor, vext::Op::SUM)->Name("BM_CudaCsrScatterTensor/SUM")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterKernel, vext::Op::MEAN)->Name("BM_CudaCsrScatterKernel/MEAN")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterTensor, vext::Op::MEAN)->Name("BM_CudaCsrScatterTensor/MEAN")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterKernel, vext::Op::MAX)->Name("BM_CudaCsrScatterKernel/MAX")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterTensor, vext::Op::MAX)->Name("BM_CudaCsrScatterTensor/MAX")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterKernel, vext::Op::MIN)->Name("BM_CudaCsrScatterKernel/MIN")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterTensor, vext::Op::MIN)->Name("BM_CudaCsrScatterTensor/MIN")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterKernel, vext::Op::PROD)->Name("BM_CudaCsrScatterKernel/PROD")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterTensor, vext::Op::PROD)->Name("BM_CudaCsrScatterTensor/PROD")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterKernel, vext::Op::STD)->Name("BM_CudaCsrScatterKernel/STD")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterTensor, vext::Op::STD)->Name("BM_CudaCsrScatterTensor/STD")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterKernel, vext::Op::VAR)->Name("BM_CudaCsrScatterKernel/VAR")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterTensor, vext::Op::VAR)->Name("BM_CudaCsrScatterTensor/VAR")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();

BENCHMARK_TEMPLATE(BM_CudaCsrSpmvKernel, vext::Op::SUM)->Name("BM_CudaCsrSpmvKernel/SUM")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrSpmvTensor, vext::Op::SUM)->Name("BM_CudaCsrSpmvTensor/SUM")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrSpmvKernel, vext::Op::MEAN)->Name("BM_CudaCsrSpmvKernel/MEAN")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrSpmvTensor, vext::Op::MEAN)->Name("BM_CudaCsrSpmvTensor/MEAN")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrSpmvKernel, vext::Op::MAX)->Name("BM_CudaCsrSpmvKernel/MAX")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrSpmvTensor, vext::Op::MAX)->Name("BM_CudaCsrSpmvTensor/MAX")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrSpmvKernel, vext::Op::MIN)->Name("BM_CudaCsrSpmvKernel/MIN")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrSpmvTensor, vext::Op::MIN)->Name("BM_CudaCsrSpmvTensor/MIN")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrSpmvKernel, vext::Op::PROD)->Name("BM_CudaCsrSpmvKernel/PROD")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrSpmvTensor, vext::Op::PROD)->Name("BM_CudaCsrSpmvTensor/PROD")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrSpmvKernel, vext::Op::STD)->Name("BM_CudaCsrSpmvKernel/STD")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrSpmvTensor, vext::Op::STD)->Name("BM_CudaCsrSpmvTensor/STD")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrSpmvKernel, vext::Op::VAR)->Name("BM_CudaCsrSpmvKernel/VAR")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrSpmvTensor, vext::Op::VAR)->Name("BM_CudaCsrSpmvTensor/VAR")->Args({ CSR_ROWS, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();

BENCHMARK(BM_CudaMatmulKernel)->Args({ MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE })->Iterations(MATMUL_ITERS)->UseManualTime();
BENCHMARK(BM_CudaTensorMatmul)->Args({ MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE })->Iterations(MATMUL_ITERS)->UseManualTime();
BENCHMARK(BM_CudaNnLinearForward)->Arg(MATRIX_SIZE)->Iterations(MATMUL_ITERS)->UseManualTime();

BENCHMARK_MAIN();

#undef VEXT_BENCHMARK_NOINLINE
