cmake_minimum_required(VERSION 3.18) # 3.18+ recommended for CUDA
project(thunderfft LANGUAGES CXX CUDA)

# ===== Build config =====
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Architectures (e.g., "80;90")
set(THUNDERFFT_ARCHS "80" CACHE STRING "CUDA SM architectures")
set(CMAKE_CUDA_ARCHITECTURES ${THUNDERFFT_ARCHS})

# Find CUDA toolkit (pass -DCUDAToolkit_DIR=... if needed)
find_package(CUDAToolkit REQUIRED)

# Header-only interface target
add_library(thunderfft INTERFACE)
target_include_directories(thunderfft INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
# Consumers need cudart
target_link_libraries(thunderfft INTERFACE CUDA::cudart)
target_compile_features(thunderfft INTERFACE cxx_std_17)

# Subdirs
add_subdirectory(sample)
#add_subdirectory(bench)
