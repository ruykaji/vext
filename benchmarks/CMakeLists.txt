cmake_minimum_required(VERSION 3.24)
project(vext_benchmarks LANGUAGES CXX)

option(VEXT_BENCHMARK_ENABLE_CUDA "Build CUDA comparisons when a CUDA toolchain is available" ON)

set(VEXT_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures used by benchmark targets")

find_package(benchmark CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

set(VEXT_BENCHMARK_CUDA_AVAILABLE OFF)

if(VEXT_BENCHMARK_ENABLE_CUDA)
	include(CheckLanguage)
	check_language(CUDA)

	if(CMAKE_CUDA_COMPILER)
		enable_language(CUDA)
		find_package(CUDAToolkit REQUIRED)
		set(VEXT_BENCHMARK_CUDA_AVAILABLE ON)
	endif()
endif()

if(VEXT_BENCHMARK_CUDA_AVAILABLE)
	find_package(vext CONFIG REQUIRED COMPONENTS cuda)
else()
	find_package(vext CONFIG REQUIRED)
endif()

function(vext_add_benchmark target source)
	add_executable(${target} ${source})
	target_include_directories(${target}
		PRIVATE
			"${CMAKE_CURRENT_SOURCE_DIR}/include"
	)
	target_compile_features(${target}
		PRIVATE
			cxx_std_20
	)
	set_target_properties(${target}
		PROPERTIES
			RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
	)
endfunction()

vext_add_benchmark(bench_compare_cpu src/compare_cpu.cpp)
target_link_libraries(bench_compare_cpu
	PRIVATE
		vext::cpu
		benchmark::benchmark
		Eigen3::Eigen
)

target_compile_definitions(bench_compare_cpu
	PRIVATE
		EIGEN_DONT_PARALLELIZE
)

if(VEXT_BENCHMARK_CUDA_AVAILABLE)
	vext_add_benchmark(bench_compare_cuda src/compare_cuda.cu)
	target_link_libraries(bench_compare_cuda
		PRIVATE
			vext::cuda
			benchmark::benchmark
			CUDA::cublas
			CUDA::cusparse
	)
	set_target_properties(bench_compare_cuda
		PROPERTIES
			CUDA_ARCHITECTURES "${VEXT_CUDA_ARCHITECTURES}"
			CUDA_STANDARD 20
			CUDA_STANDARD_REQUIRED ON
	)

else()
	message(STATUS "Skipping CUDA benchmarks because a CUDA compiler is unavailable")
endif()

message(STATUS "===================================================")
message(STATUS "Project Name            : ${PROJECT_NAME}")
message(STATUS "CMake Version           : ${CMAKE_VERSION}")
message(STATUS "CXX Compiler            : ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA comparisons        : ${VEXT_BENCHMARK_CUDA_AVAILABLE}")
message(STATUS "Build Type              : ${CMAKE_BUILD_TYPE}")
message(STATUS "===================================================\n")
