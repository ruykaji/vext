add_library(vext
    STATIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpu/allocator.cpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpu/operations.cpp>
)
