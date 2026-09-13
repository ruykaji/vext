if(VEXT_BUILD_TESTS)
	include(add_gtest)

	add_gtest(test_ops_cpu cpu/ops.cpp vext::cpu)
	add_gtest(test_tensor_cpu cpu/tensor.cpp vext::cpu)
	add_gtest(test_nn_cpu cpu/nn.cpp vext::cpu)

	if(TARGET vext::cuda)
		add_gtest(test_ops_cuda cuda/ops.cu vext::cuda)
		add_gtest(test_tensor_cuda cuda/tensor.cu vext::cuda)
		add_gtest(test_nn_cuda cuda/nn.cu vext::cuda)

		set_target_properties(test_ops_cuda test_tensor_cuda test_nn_cuda
            PROPERTIES
			    CUDA_ARCHITECTURES "${VEXT_CUDA_ARCHITECTURES}"
			    CUDA_STANDARD 20
			    CUDA_STANDARD_REQUIRED ON
            )
	else()
		message(STATUS "Skipping CUDA tests because vext::cuda is unavailable")
	endif()
endif()
