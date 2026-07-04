cmake_minimum_required(VERSION 3.15.0)
project(vext VERSION 0.0.1 LANGUAGES C CXX)

option(VEXT_BUILD_TEST "Build Test" ON)
option(VEXT_ENABLE_CUDA "Enable cuda" ON)

set(VEXT_CUDA OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(VEXT_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit QUIET)

        if(CUDAToolkit_FOUND)
            set(VEXT_CUDA ON)

            message(STATUS "CUDA backend enabled")
            message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
            message(STATUS "CUDA toolkit: ${CUDAToolkit_VERSION}")
        else()
            message(STATUS "CUDA compiler found, but CUDA Toolkit not found. Building CPU-only.")
        endif()
    else()
        message(STATUS "CUDA compiler not found. Building CPU-only.")
    endif()
endif()

add_library(vext
    INTERFACE)

target_compile_definitions(vext 
   INTERFACE 
        $<$<CONFIG:Debug>:DEBUG=1>
        VEXT_CUDA=$<BOOL:${VEXT_CUDA}>
)

target_include_directories(vext
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(VEXT_CUDA)
    set_target_properties(vext PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
    )

    target_link_libraries(vext
        INTERFACE
            CUDA::cudart
    )
endif()

if(VEXT_BUILD_TEST)
    find_package(GTest)

    if(GTEST_FOUND)
        enable_testing()

        include(GoogleTest)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
    else()
        message(WARNING "Failed to find gtest package")
    endif()
endif()

message(STATUS "===================================================")
message(STATUS "Project Name  : ${PROJECT_NAME}")
message(STATUS "CMake Version : ${CMAKE_VERSION}")
message(STATUS "C++ Standard  : ${CMAKE_CXX_STANDARD}")
message(STATUS "CXX Compiler  : ${CMAKE_CXX_COMPILER}")
message(STATUS "Build Test    : ${VEXT_BUILD_TEST}")
message(STATUS "Build Type    : ${CMAKE_BUILD_TYPE}")
message(STATUS "Debug Flags   : ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "Release Flags : ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "===================================================\n")
