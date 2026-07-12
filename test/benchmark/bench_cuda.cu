#include <benchmark/benchmark.h>

#include <limits>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include <vext/core/cuda/allocator.cuh>
#include <vext/core/cuda/operations/csr_scatter.cuh>
#include <vext/core/cuda/operations/elementwise_binary.cuh>
#include <vext/core/cuda/operations/elementwise_logical.cuh>
#include <vext/core/cuda/operations/elementwise_unary.cuh>
#include <vext/core/cuda/operations/linear_algebra.cuh>
#include <vext/core/cuda/operations/memory.cuh>
#include <vext/core/cuda/operations/reduction.cuh>
#include <vext/nn/layer/linear.hpp>
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
	vext::core::cuda::operations::memcpy<Tp, vext::Backend::CUDA, vext::Backend::CPU>(device, host.data(), static_cast<std::uint32_t>(host.size()));
	return device;
}

template <typename Tp>
VEXT_BENCHMARK_NOINLINE void
observe_tensor(
	const vext::Tensor<Tp, vext::Backend::CUDA>& tensor)
{
	benchmark::DoNotOptimize(tensor.shape().length());
}

template <vext::core::BinaryOperation Kp>
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
			vext::core::cuda::operations::binary<Kp>(out, lhs, rhs, size);
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

template <vext::core::BinaryOperation Kp>
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

				if constexpr(Kp == vext::core::BinaryOperation::ADD)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = lhs + rhs;
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::BinaryOperation::SUB)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = lhs - rhs;
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::BinaryOperation::MUL)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = lhs * rhs;
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::BinaryOperation::DIV)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = lhs / rhs;
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::BinaryOperation::POW)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = lhs ^ rhs;
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::BinaryOperation::PRELU)
					{
						vext::Tensor<float, vext::Backend::CUDA> out(lhs);
						out.prelu(rhs);
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

template <vext::core::BinaryOperation Kp>
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

				if constexpr(Kp == vext::core::BinaryOperation::ADD)
					{
						out += rhs;
					}
				else if constexpr(Kp == vext::core::BinaryOperation::SUB)
					{
						out -= rhs;
					}
				else if constexpr(Kp == vext::core::BinaryOperation::MUL)
					{
						out *= rhs;
					}
				else if constexpr(Kp == vext::core::BinaryOperation::DIV)
					{
						out /= rhs;
					}
				else if constexpr(Kp == vext::core::BinaryOperation::POW)
					{
						out ^= rhs;
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

template <vext::core::LogicOperation Kp>
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
			vext::core::cuda::operations::logical<Kp>(out, lhs, rhs, size);
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

template <vext::core::LogicOperation Kp>
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

				if constexpr(Kp == vext::core::LogicOperation::EQUAL)
					{
						const vext::Tensor<std::uint8_t, vext::Backend::CUDA> out = lhs == rhs;
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::LogicOperation::NOT_EQUAL)
					{
						const vext::Tensor<std::uint8_t, vext::Backend::CUDA> out = lhs != rhs;
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::LogicOperation::LESS)
					{
						const vext::Tensor<std::uint8_t, vext::Backend::CUDA> out = lhs < rhs;
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::LogicOperation::LESS_EQUAL)
					{
						const vext::Tensor<std::uint8_t, vext::Backend::CUDA> out = lhs <= rhs;
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::LogicOperation::GREATER)
					{
						const vext::Tensor<std::uint8_t, vext::Backend::CUDA> out = lhs > rhs;
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::LogicOperation::GREATER_EQUAL)
					{
						const vext::Tensor<std::uint8_t, vext::Backend::CUDA> out = lhs >= rhs;
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

template <vext::core::UnaryOperation Kp>
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

			if constexpr(Kp == vext::core::UnaryOperation::LEAKY_RELU || Kp == vext::core::UnaryOperation::ELU || Kp == vext::core::UnaryOperation::SWISH)
				{
					vext::core::cuda::operations::unary<Kp>(values, size, 0.25f);
				}
			else if constexpr(Kp == vext::core::UnaryOperation::LINEAR || Kp == vext::core::UnaryOperation::CLIP || Kp == vext::core::UnaryOperation::POW)
				{
					vext::core::cuda::operations::unary<Kp>(values, size, 0.75f, 1.25f);
				}
			else
				{
					vext::core::cuda::operations::unary<Kp>(values, size);
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

template <vext::core::UnaryOperation Kp>
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

				if constexpr(Kp == vext::core::UnaryOperation::ABS)
					{
						out.abs();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::SIN)
					{
						out.sin();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::COS)
					{
						out.cos();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::NEG)
					{
						-out;
					}
				else if constexpr(Kp == vext::core::UnaryOperation::EXP)
					{
						out.exp();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::LOG)
					{
						out.log();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::SQRT)
					{
						out.sqrt();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::SQUARE)
					{
						out.square();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::ROUND)
					{
						out.round();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::SIGMOID)
					{
						out.sigmoid();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::SOFT_RELU)
					{
						out.soft_relu();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::RELU)
					{
						out.relu();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::SOFTMAX)
					{
						out.softmax();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::SOFTMIN)
					{
						out.softmin();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::LOGSOFTMAX)
					{
						out.log_softmax();
					}
				else if constexpr(Kp == vext::core::UnaryOperation::LEAKY_RELU)
					{
						out.leaky_relu(0.25f);
					}
				else if constexpr(Kp == vext::core::UnaryOperation::ELU)
					{
						out.elu(0.25f);
					}
				else if constexpr(Kp == vext::core::UnaryOperation::SWISH)
					{
						out.swish(0.25f);
					}
				else if constexpr(Kp == vext::core::UnaryOperation::LINEAR)
					{
						out.linear(0.75f, 1.25f);
					}
				else if constexpr(Kp == vext::core::UnaryOperation::CLIP)
					{
						out.clip(0.25f, 0.75f);
					}
				else if constexpr(Kp == vext::core::UnaryOperation::POW)
					{
						out.pow(0.75f, 1.25f);
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

template <vext::core::ReductionOperation Kp>
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
			vext::core::cuda::operations::reduce<Kp>(out, values, 1, size, keep_dims, keep_strides, reduce_dims, reduce_strides);
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

template <vext::core::ReductionOperation Kp>
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

				if constexpr(Kp == vext::core::ReductionOperation::SUM)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = tensor.sum();
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::ReductionOperation::MEAN)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = tensor.mean();
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::ReductionOperation::MAX)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = tensor.max();
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::ReductionOperation::MIN)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = tensor.min();
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::ReductionOperation::PROD)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = tensor.prod();
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::ReductionOperation::STD)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = tensor.std();
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::ReductionOperation::VAR)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = tensor.var();
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

template <vext::core::CSRScatterOperation Kp>
std::vector<float>
make_csr_output_seed(
	const std::uint32_t size)
{
	float value = 0.0f;

	if constexpr(Kp == vext::core::CSRScatterOperation::MIN)
		{
			value = std::numeric_limits<float>::max();
		}
	else if constexpr(Kp == vext::core::CSRScatterOperation::MAX)
		{
			value = std::numeric_limits<float>::lowest();
		}
	else if constexpr(Kp == vext::core::CSRScatterOperation::PROD)
		{
			value = 1.0f;
		}

	std::vector<float> seed(size, value);
	return seed;
}

template <vext::core::CSRScatterOperation Kp>
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
			vext::core::cuda::operations::csr_scatter<Kp>(out, src, head, tail, rows, features);
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

template <vext::core::CSRScatterOperation Kp>
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

				if constexpr(Kp == vext::core::CSRScatterOperation::SUM)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = src.csr_scatter_sum(head, tail);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::CSRScatterOperation::MEAN)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = src.csr_scatter_mean(head, tail);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::CSRScatterOperation::MAX)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = src.csr_scatter_max(head, tail);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::CSRScatterOperation::MIN)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = src.csr_scatter_min(head, tail);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::CSRScatterOperation::PROD)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = src.csr_scatter_prod(head, tail);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::CSRScatterOperation::STD)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = src.csr_scatter_std(head, tail);
						state.SetIterationTime(timer.stop_seconds());
						observe_tensor(out);
					}
				else if constexpr(Kp == vext::core::CSRScatterOperation::VAR)
					{
						const vext::Tensor<float, vext::Backend::CUDA> out = src.csr_scatter_var(head, tail);
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
			vext::core::cuda::operations::matmul(out, lhs, rhs, m, p, n);
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
				const vext::Tensor<float, vext::Backend::CUDA> out = lhs.matmul(rhs);
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

BENCHMARK_TEMPLATE(BM_CudaBinaryKernel, vext::core::BinaryOperation::ADD)->Name("BM_CudaBinaryKernel/ADD")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensor, vext::core::BinaryOperation::ADD)->Name("BM_CudaBinaryTensor/ADD")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensorInPlace, vext::core::BinaryOperation::ADD)->Name("BM_CudaBinaryTensorInPlace/ADD")->Args({ INPLACE_ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryKernel, vext::core::BinaryOperation::SUB)->Name("BM_CudaBinaryKernel/SUB")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensor, vext::core::BinaryOperation::SUB)->Name("BM_CudaBinaryTensor/SUB")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensorInPlace, vext::core::BinaryOperation::SUB)->Name("BM_CudaBinaryTensorInPlace/SUB")->Args({ INPLACE_ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryKernel, vext::core::BinaryOperation::MUL)->Name("BM_CudaBinaryKernel/MUL")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensor, vext::core::BinaryOperation::MUL)->Name("BM_CudaBinaryTensor/MUL")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensorInPlace, vext::core::BinaryOperation::MUL)->Name("BM_CudaBinaryTensorInPlace/MUL")->Args({ INPLACE_ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryKernel, vext::core::BinaryOperation::DIV)->Name("BM_CudaBinaryKernel/DIV")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensor, vext::core::BinaryOperation::DIV)->Name("BM_CudaBinaryTensor/DIV")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensorInPlace, vext::core::BinaryOperation::DIV)->Name("BM_CudaBinaryTensorInPlace/DIV")->Args({ INPLACE_ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryKernel, vext::core::BinaryOperation::POW)->Name("BM_CudaBinaryKernel/POW")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensor, vext::core::BinaryOperation::POW)->Name("BM_CudaBinaryTensor/POW")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensorInPlace, vext::core::BinaryOperation::POW)->Name("BM_CudaBinaryTensorInPlace/POW")->Args({ INPLACE_ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryKernel, vext::core::BinaryOperation::PRELU)->Name("BM_CudaBinaryKernel/PRELU")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryTensor, vext::core::BinaryOperation::PRELU)->Name("BM_CudaBinaryTensor/PRELU")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryKernel, vext::core::BinaryOperation::MIN)->Name("BM_CudaBinaryKernel/MIN/kernel_only")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaBinaryKernel, vext::core::BinaryOperation::MAX)->Name("BM_CudaBinaryKernel/MAX/kernel_only")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();

BENCHMARK_TEMPLATE(BM_CudaLogicalKernel, vext::core::LogicOperation::EQUAL)->Name("BM_CudaLogicalKernel/EQUAL")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalTensor, vext::core::LogicOperation::EQUAL)->Name("BM_CudaLogicalTensor/EQUAL")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalKernel, vext::core::LogicOperation::NOT_EQUAL)->Name("BM_CudaLogicalKernel/NOT_EQUAL")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalTensor, vext::core::LogicOperation::NOT_EQUAL)->Name("BM_CudaLogicalTensor/NOT_EQUAL")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalKernel, vext::core::LogicOperation::LESS)->Name("BM_CudaLogicalKernel/LESS")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalTensor, vext::core::LogicOperation::LESS)->Name("BM_CudaLogicalTensor/LESS")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalKernel, vext::core::LogicOperation::LESS_EQUAL)->Name("BM_CudaLogicalKernel/LESS_EQUAL")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalTensor, vext::core::LogicOperation::LESS_EQUAL)->Name("BM_CudaLogicalTensor/LESS_EQUAL")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalKernel, vext::core::LogicOperation::GREATER)->Name("BM_CudaLogicalKernel/GREATER")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalTensor, vext::core::LogicOperation::GREATER)->Name("BM_CudaLogicalTensor/GREATER")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalKernel, vext::core::LogicOperation::GREATER_EQUAL)->Name("BM_CudaLogicalKernel/GREATER_EQUAL")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaLogicalTensor, vext::core::LogicOperation::GREATER_EQUAL)->Name("BM_CudaLogicalTensor/GREATER_EQUAL")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();

BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::ABS)->Name("BM_CudaUnaryKernel/ABS")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::ABS)->Name("BM_CudaUnaryTensor/ABS")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::SIN)->Name("BM_CudaUnaryKernel/SIN")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::SIN)->Name("BM_CudaUnaryTensor/SIN")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::COS)->Name("BM_CudaUnaryKernel/COS")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::COS)->Name("BM_CudaUnaryTensor/COS")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::TANH)->Name("BM_CudaUnaryKernel/TANH/kernel_only")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::NEG)->Name("BM_CudaUnaryKernel/NEG")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::NEG)->Name("BM_CudaUnaryTensor/NEG")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::EXP)->Name("BM_CudaUnaryKernel/EXP")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::EXP)->Name("BM_CudaUnaryTensor/EXP")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::LOG)->Name("BM_CudaUnaryKernel/LOG")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::LOG)->Name("BM_CudaUnaryTensor/LOG")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::SQRT)->Name("BM_CudaUnaryKernel/SQRT")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::SQRT)->Name("BM_CudaUnaryTensor/SQRT")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::SQUARE)->Name("BM_CudaUnaryKernel/SQUARE")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::SQUARE)->Name("BM_CudaUnaryTensor/SQUARE")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::ROUND)->Name("BM_CudaUnaryKernel/ROUND")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::ROUND)->Name("BM_CudaUnaryTensor/ROUND")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::SIGMOID)->Name("BM_CudaUnaryKernel/SIGMOID")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::SIGMOID)->Name("BM_CudaUnaryTensor/SIGMOID")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::SOFT_RELU)->Name("BM_CudaUnaryKernel/SOFT_RELU")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::SOFT_RELU)->Name("BM_CudaUnaryTensor/SOFT_RELU")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::RELU)->Name("BM_CudaUnaryKernel/RELU")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::RELU)->Name("BM_CudaUnaryTensor/RELU")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::SOFTMAX)->Name("BM_CudaUnaryKernel/SOFTMAX")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::SOFTMAX)->Name("BM_CudaUnaryTensor/SOFTMAX")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::SOFTMIN)->Name("BM_CudaUnaryKernel/SOFTMIN")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::SOFTMIN)->Name("BM_CudaUnaryTensor/SOFTMIN")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::LOGSOFTMAX)->Name("BM_CudaUnaryKernel/LOGSOFTMAX")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::LOGSOFTMAX)->Name("BM_CudaUnaryTensor/LOGSOFTMAX")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::LEAKY_RELU)->Name("BM_CudaUnaryKernel/LEAKY_RELU")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::LEAKY_RELU)->Name("BM_CudaUnaryTensor/LEAKY_RELU")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::ELU)->Name("BM_CudaUnaryKernel/ELU")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::ELU)->Name("BM_CudaUnaryTensor/ELU")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::SWISH)->Name("BM_CudaUnaryKernel/SWISH")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::SWISH)->Name("BM_CudaUnaryTensor/SWISH")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::LINEAR)->Name("BM_CudaUnaryKernel/LINEAR")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::LINEAR)->Name("BM_CudaUnaryTensor/LINEAR")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::CLIP)->Name("BM_CudaUnaryKernel/CLIP")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::CLIP)->Name("BM_CudaUnaryTensor/CLIP")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryKernel, vext::core::UnaryOperation::POW)->Name("BM_CudaUnaryKernel/UNARY_POW")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaUnaryTensor, vext::core::UnaryOperation::POW)->Name("BM_CudaUnaryTensor/UNARY_POW")->Args({ ELEMENT_COUNT, 1 })->Iterations(ELEMENT_ITERS)->UseManualTime();

BENCHMARK_TEMPLATE(BM_CudaReductionKernel, vext::core::ReductionOperation::SUM)->Name("BM_CudaReductionKernel/SUM")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionTensor, vext::core::ReductionOperation::SUM)->Name("BM_CudaReductionTensor/SUM")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionKernel, vext::core::ReductionOperation::MEAN)->Name("BM_CudaReductionKernel/MEAN")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionTensor, vext::core::ReductionOperation::MEAN)->Name("BM_CudaReductionTensor/MEAN")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionKernel, vext::core::ReductionOperation::MAX)->Name("BM_CudaReductionKernel/MAX")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionTensor, vext::core::ReductionOperation::MAX)->Name("BM_CudaReductionTensor/MAX")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionKernel, vext::core::ReductionOperation::MIN)->Name("BM_CudaReductionKernel/MIN")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionTensor, vext::core::ReductionOperation::MIN)->Name("BM_CudaReductionTensor/MIN")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionKernel, vext::core::ReductionOperation::PROD)->Name("BM_CudaReductionKernel/PROD")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionTensor, vext::core::ReductionOperation::PROD)->Name("BM_CudaReductionTensor/PROD")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionKernel, vext::core::ReductionOperation::STD)->Name("BM_CudaReductionKernel/STD")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionTensor, vext::core::ReductionOperation::STD)->Name("BM_CudaReductionTensor/STD")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionKernel, vext::core::ReductionOperation::VAR)->Name("BM_CudaReductionKernel/VAR")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaReductionTensor, vext::core::ReductionOperation::VAR)->Name("BM_CudaReductionTensor/VAR")->Arg(ELEMENT_COUNT)->Iterations(ELEMENT_ITERS)->UseManualTime();

BENCHMARK_TEMPLATE(BM_CudaCsrScatterKernel, vext::core::CSRScatterOperation::SUM)->Name("BM_CudaCsrScatterKernel/SUM")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterTensor, vext::core::CSRScatterOperation::SUM)->Name("BM_CudaCsrScatterTensor/SUM")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterKernel, vext::core::CSRScatterOperation::MEAN)->Name("BM_CudaCsrScatterKernel/MEAN")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterTensor, vext::core::CSRScatterOperation::MEAN)->Name("BM_CudaCsrScatterTensor/MEAN")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterKernel, vext::core::CSRScatterOperation::MAX)->Name("BM_CudaCsrScatterKernel/MAX")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterTensor, vext::core::CSRScatterOperation::MAX)->Name("BM_CudaCsrScatterTensor/MAX")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterKernel, vext::core::CSRScatterOperation::MIN)->Name("BM_CudaCsrScatterKernel/MIN")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterTensor, vext::core::CSRScatterOperation::MIN)->Name("BM_CudaCsrScatterTensor/MIN")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterKernel, vext::core::CSRScatterOperation::PROD)->Name("BM_CudaCsrScatterKernel/PROD")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterTensor, vext::core::CSRScatterOperation::PROD)->Name("BM_CudaCsrScatterTensor/PROD")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterKernel, vext::core::CSRScatterOperation::STD)->Name("BM_CudaCsrScatterKernel/STD")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterTensor, vext::core::CSRScatterOperation::STD)->Name("BM_CudaCsrScatterTensor/STD")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterKernel, vext::core::CSRScatterOperation::VAR)->Name("BM_CudaCsrScatterKernel/VAR")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();
BENCHMARK_TEMPLATE(BM_CudaCsrScatterTensor, vext::core::CSRScatterOperation::VAR)->Name("BM_CudaCsrScatterTensor/VAR")->Args({ CSR_ROWS, CSR_FEATURES, CSR_DEGREE })->Iterations(ELEMENT_ITERS)->UseManualTime();

BENCHMARK(BM_CudaMatmulKernel)->Args({ MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE })->Iterations(MATMUL_ITERS)->UseManualTime();
BENCHMARK(BM_CudaTensorMatmul)->Args({ MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE })->Iterations(MATMUL_ITERS)->UseManualTime();
BENCHMARK(BM_CudaNnLinearForward)->Arg(MATRIX_SIZE)->Iterations(MATMUL_ITERS)->UseManualTime();

BENCHMARK_MAIN();

#undef VEXT_BENCHMARK_NOINLINE
