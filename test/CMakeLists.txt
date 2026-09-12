if(VEXT_BUILD_TESTS)
	include(add_gtest)

	add_gtest(test_ops_cpu cpu/ops.cpp vext::cpu)
	add_gtest(test_tensor_cpu cpu/tensor.cpp vext::cpu)
	add_gtest(test_nn_cpu cpu/nn.cpp vext::cpu)

	if(TARGET vext::cuda)
		add_gtest(test_ops_cuda cuda/ops.cu vext::cuda)
		add_gtest(test_tensor_cuda cuda/tensor.cu vext::cuda)
		add_gtest(test_nn_cuda cuda/nn.cu vext::cuda)

		set_target_properties(test_ops_cuda test_tensor_cuda test_nn_cuda PROPERTIES
			CUDA_ARCHITECTURES "${VEXT_CUDA_ARCHITECTURES}"
			CUDA_STANDARD 20
			CUDA_STANDARD_REQUIRED ON)
	else()
		message(STATUS "Skipping CUDA tests because vext::cuda is unavailable")
	endif()
endif()

if(VEXT_BUILD_BENCHMARKS)
	add_executable(bench_cpu benchmark/bench_cpu.cpp)
	target_link_libraries(bench_cpu PRIVATE vext::cpu benchmark::benchmark)
	vext_apply_project_options(bench_cpu)

	if(TARGET vext::cuda)
		add_executable(bench_cuda benchmark/bench_cuda.cu)
		target_link_libraries(bench_cuda PRIVATE vext::cuda benchmark::benchmark)
		set_target_properties(bench_cuda PROPERTIES
			CUDA_ARCHITECTURES "${VEXT_CUDA_ARCHITECTURES}"
			CUDA_STANDARD 20
			CUDA_STANDARD_REQUIRED ON)
		vext_apply_project_options(bench_cuda NO_SANITIZERS)
	else()
		message(STATUS "Skipping CUDA benchmarks because vext::cuda is unavailable")
	endif()
endif()
