#ifndef __VEXT_BENCHMARKS_CUDA_TIMING_CUH__
#define __VEXT_BENCHMARKS_CUDA_TIMING_CUH__

#include <benchmark/benchmark.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

namespace vext::benchmarks::cuda
{

inline void
check_cuda(
	const cudaError_t status,
	const char*       operation)
{
	if(status != cudaSuccess)
		{
			throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
		}
}

inline void
check_cublas(
	const cublasStatus_t status,
	const char*          operation)
{
	if(status != CUBLAS_STATUS_SUCCESS)
		{
			throw std::runtime_error(std::string(operation) + " failed with cuBLAS status " + std::to_string(status) + ".");
		}
}

inline void
check_cusparse(
	const cusparseStatus_t status,
	const char*            operation)
{
	if(status != CUSPARSE_STATUS_SUCCESS)
		{
			throw std::runtime_error(std::string(operation) + " failed with cuSPARSE status " + std::to_string(status) + ".");
		}
}

inline bool
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

	CudaEventTimer(const CudaEventTimer&) = delete;
	CudaEventTimer&
	operator=(const CudaEventTimer&) = delete;

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

	CublasHandle(const CublasHandle&) = delete;
	CublasHandle&
	operator=(const CublasHandle&) = delete;

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

	CusparseHandle(const CusparseHandle&) = delete;
	CusparseHandle&
	operator=(const CusparseHandle&) = delete;

	operator cusparseHandle_t() const noexcept
	{
		return __handle;
	}

private:
	cusparseHandle_t __handle = nullptr;
};

}

#endif
