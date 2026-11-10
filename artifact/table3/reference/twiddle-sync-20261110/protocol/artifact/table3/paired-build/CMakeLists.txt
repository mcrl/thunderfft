cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES "${THUNDERFFT_ARCHS}")
project(thff_paired_native LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)
if(NOT IS_DIRECTORY "${THFF_SOURCE_ROOT}/include")
  message(FATAL_ERROR "THFF_SOURCE_ROOT must contain the selected source snapshot")
endif()
add_executable(thff_paper_fft "${THFF_SOURCE_ROOT}/sample/bench/paper_fft.cu")
target_include_directories(thff_paper_fft PRIVATE
  "${THFF_SOURCE_ROOT}/include"
  "${THFF_SOURCE_ROOT}/sample/bench/util"
  "${THFF_SOURCE_ROOT}/sample/bench/baseline")
target_link_libraries(thff_paper_fft PRIVATE CUDA::cudart CUDA::cufft)
target_compile_options(thff_paper_fft PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>)
if(THFF_BUILD_VALIDATION)
  add_executable(thff_validate "${THFF_SOURCE_ROOT}/sample/validation/fft_validation.cu")
  target_include_directories(thff_validate PRIVATE "${THFF_SOURCE_ROOT}/include")
  target_link_libraries(thff_validate PRIVATE CUDA::cudart CUDA::cufft)
  target_compile_options(thff_validate PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()
