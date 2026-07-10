include("add_gtest")

add_gtest(test_shape_cpu cpu/shape.cpp vext)
add_gtest(test_core_cpu cpu/core_operations.cpp vext)
add_gtest(test_tensor_cpu cpu/tensor.cpp vext)
add_gtest(test_nn_cpu cpu/nn.cpp vext)

target_compile_options(test_tensor_cpu PRIVATE -UVEXT_CUDA)
target_compile_options(test_nn_cpu PRIVATE -UVEXT_CUDA)

if(VEXT_CUDA)
    add_gtest(test_core_cuda cuda/core_operations.cu vext)
    set_target_properties(test_core_cuda PROPERTIES CUDA_ARCHITECTURES native)

    add_gtest(test_tensor_cuda cuda/tensor.cu vext)
    set_target_properties(test_tensor_cuda PROPERTIES CUDA_ARCHITECTURES native)

    add_gtest(test_nn_cuda cuda/nn.cu vext)
    set_target_properties(test_nn_cuda PROPERTIES CUDA_ARCHITECTURES native)
else()
    message(STATUS "Skipping CUDA tests because VEXT_CUDA is OFF")
endif()

add_executable(bench_cpu benchmark/bench_cpu.cpp)
target_link_libraries(bench_cpu PRIVATE vext benchmark::benchmark)
target_compile_options(bench_cpu PRIVATE -UVEXT_CUDA)

if(VEXT_CUDA)
    add_executable(bench_cuda benchmark/bench_cuda.cu)
    target_link_libraries(bench_cuda PRIVATE vext benchmark::benchmark)
    set_target_properties(bench_cuda PROPERTIES CUDA_ARCHITECTURES native)
else()
    message(STATUS "Skipping CUDA benchmarks because VEXT_CUDA is OFF")
endif()
