#ifndef __VEXT_BENCHMARKS_CUDA_BUFFER_CUH__
#define __VEXT_BENCHMARKS_CUDA_BUFFER_CUH__

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include <cuda/timing.cuh>

namespace vext::benchmarks::cuda
{

template <typename Tp>
class DeviceBuffer
{
public:
	explicit DeviceBuffer(
		const std::uint64_t size)
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
		check_cuda(cudaMemcpy(__data, values.data(), __size * sizeof(Tp), cudaMemcpyHostToDevice),
					  "Copying benchmark data to CUDA");
	}

private:
	Tp*           __data = nullptr;
	std::uint64_t __size = 0;
};

} // namespace vext::benchmarks::cuda

#endif
