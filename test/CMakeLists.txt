include("add_gtest")

add_gtest(test_tensor_cpu cpu/tensor.cpp vext)
target_include_directories(test_tensor_cpu
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_gtest(test_nn_cpu cpu/nn.cpp vext)
target_include_directories(test_nn_cpu
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)
