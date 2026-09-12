cmake_minimum_required(VERSION 3.24)

project(vext_consumer LANGUAGES CXX)

set(VEXT_CONSUMER_BACKEND "cpu" CACHE STRING "vext backend to consume (cpu or cuda)")

if(VEXT_CONSUMER_BACKEND STREQUAL "cuda")
	enable_language(CUDA)
	find_package(vext REQUIRED COMPONENTS cuda)
	add_executable(vext_consumer main.cu)
	target_link_libraries(vext_consumer PRIVATE vext::cuda)
	set_target_properties(vext_consumer PROPERTIES
		CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
		CUDA_STANDARD 20
		CUDA_STANDARD_REQUIRED ON)
elseif(VEXT_CONSUMER_BACKEND STREQUAL "cpu")
	find_package(vext REQUIRED)
	add_executable(vext_consumer main.cpp)
	target_link_libraries(vext_consumer PRIVATE vext::cpu)
else()
	message(FATAL_ERROR "VEXT_CONSUMER_BACKEND must be cpu or cuda")
endif()
