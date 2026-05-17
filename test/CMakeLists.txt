include("add_gtest")

add_gtest(test_allocator_cpu allocator_cpu.cpp vext)
target_include_directories(test_allocator_cpu
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)
