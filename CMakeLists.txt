cmake_minimum_required(VERSION 3.24)

project(vext VERSION 0.0.1 LANGUAGES CXX)

option(VEXT_ENABLE_CUDA        "Build the CUDA backend when a CUDA toolchain is available" ON)
option(VEXT_BUILD_TESTS        "Build the test suite"                                      ON)
option(VEXT_BUILD_BENCHMARKS   "Build benchmarks"                                          ON)
option(VEXT_ENABLE_SANITIZERS  "Enable AddressSanitizer and UndefinedBehaviorSanitizer"    OFF)
option(VEXT_WARNINGS_AS_ERRORS "Treat compiler warnings as errors"                         OFF)

set(VEXT_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures used by vext CUDA targets")

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(project_options)

add_library(vext_cpu INTERFACE)
add_library(vext::cpu 
    ALIAS 
        vext_cpu
)

set_target_properties(vext_cpu 
    PROPERTIES 
        EXPORT_NAME cpu
)
target_compile_features(vext_cpu 
    INTERFACE 
        cxx_std_20
)
target_compile_definitions(vext_cpu 
    INTERFACE
	    VEXT_CUDA=0
	    $<$<CONFIG:Debug>:DEBUG=1>
)
target_include_directories(vext_cpu 
    INTERFACE
	    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set(VEXT_CUDA_AVAILABLE OFF)

if(VEXT_ENABLE_CUDA)
	include(CheckLanguage)
	check_language(CUDA)
	
    if(CMAKE_CUDA_COMPILER)
		enable_language(CUDA)
		find_package(CUDAToolkit QUIET)
		
        if(CUDAToolkit_FOUND)
			set(VEXT_CUDA_AVAILABLE ON)

			add_library(vext_cuda INTERFACE)
			add_library(vext::cuda 
                ALIAS 
                    vext_cuda
            )
			
            set_target_properties(vext_cuda 
                PROPERTIES 
                    EXPORT_NAME cuda
            )
            target_compile_features(vext_cuda 
                INTERFACE 
                    cxx_std_20
            )
			target_compile_definitions(vext_cuda 
                INTERFACE
				    VEXT_CUDA=1
				    $<$<CONFIG:Debug>:DEBUG=1>
            )
			target_include_directories(vext_cuda 
                INTERFACE
				    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
				    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
            )
			target_link_libraries(vext_cuda INTERFACE CUDA::cudart)
			
            message(STATUS "CUDA backend enabled (${CUDAToolkit_VERSION})")
		else()
			message(STATUS "CUDA compiler found, but CUDAToolkit was not found; CUDA targets are disabled")
		endif()
	else()
		message(STATUS "CUDA compiler was not found; CUDA targets are disabled")
	endif()
endif()

if(VEXT_BUILD_TESTS)
	find_package(GTest REQUIRED)
endif()

if(VEXT_BUILD_BENCHMARKS)
	find_package(benchmark REQUIRED)
endif()

if(VEXT_BUILD_TESTS OR VEXT_BUILD_BENCHMARKS)
	add_subdirectory(test)
endif()

install(
    DIRECTORY 
        include/ 
    DESTINATION 
        "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(
    TARGETS 
        vext_cpu 
    EXPORT 
        vextCpuTargets
)
install(
    EXPORT
        vextCpuTargets
	FILE 
        vextCpuTargets.cmake
	NAMESPACE 
        vext::
	DESTINATION 
        "${CMAKE_INSTALL_LIBDIR}/cmake/vext"
)

if(VEXT_CUDA_AVAILABLE)
	install(
        TARGETS 
            vext_cuda 
        EXPORT 
            vextCudaTargets
    )
	install(
        EXPORT 
            vextCudaTargets
		FILE 
            vextCudaTargets.cmake
		NAMESPACE 
            vext::
		DESTINATION 
            "${CMAKE_INSTALL_LIBDIR}/cmake/vext"
    )
endif()

configure_package_config_file(
	cmake/vextConfig.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}/vextConfig.cmake"
	INSTALL_DESTINATION 
        "${CMAKE_INSTALL_LIBDIR}/cmake/vext"
)
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/vextConfigVersion.cmake"
	VERSION 
        "${PROJECT_VERSION}"
	COMPATIBILITY 
        SameMajorVersion
)
install(FILES
	"${CMAKE_CURRENT_BINARY_DIR}/vextConfig.cmake"
	"${CMAKE_CURRENT_BINARY_DIR}/vextConfigVersion.cmake"
	DESTINATION 
        "${CMAKE_INSTALL_LIBDIR}/cmake/vext"
)

message(STATUS "===================================================")
message(STATUS "Project Name     : ${PROJECT_NAME}")
message(STATUS "CMake Version    : ${CMAKE_VERSION}")
message(STATUS "C++ Standard     : ${CMAKE_CXX_STANDARD}")
message(STATUS "CXX Compiler     : ${CMAKE_CXX_COMPILER}")
message(STATUS "Build tests      : ${VEXT_BUILD_TESTS}")
message(STATUS "Build benchmarks : ${VEXT_BUILD_BENCHMARKS}")
message(STATUS "Build Type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "Debug Flags      : ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "Release Flags    : ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "===================================================\n")
