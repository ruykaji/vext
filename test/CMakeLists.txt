include("add_gtest")

add_gtest(test_allocator_cpu allocator_cpu.cpp vext)
target_include_directories(test_allocator_cpu
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_gtest(test_operations_cpu operations_cpu.cpp vext)
target_include_directories(test_operations_cpu
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_gtest(test_tensor tensor.cpp vext)
target_include_directories(test_tensor
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)
