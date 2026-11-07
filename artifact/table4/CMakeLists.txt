cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures for Table 4")
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(thunderfft_ideal_mma LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
add_executable(thff_ideal_mma ideal_mma.cu)
target_include_directories(thff_ideal_mma PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_link_libraries(thff_ideal_mma PRIVATE CUDA::cudart)
set_target_properties(thff_ideal_mma PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_compile_options(thff_ideal_mma PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)
