# bench/CMakeLists.txt

add_executable(benchmark
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/baseline/baseline_cufft.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/util/stat.cu
)

target_include_directories(benchmark PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/util
  ${CMAKE_CURRENT_SOURCE_DIR}/baseline
)

target_link_libraries(benchmark PRIVATE
  thunderfft
  CUDA::cudart
  CUDA::cufft
)


# 공통 최적화 옵션
target_compile_options(benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(thff_paper_fft paper_fft.cu)
target_include_directories(thff_paper_fft PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/util
  ${CMAKE_CURRENT_SOURCE_DIR}/baseline
)
target_link_libraries(thff_paper_fft PRIVATE thunderfft CUDA::cudart CUDA::cufft)
target_compile_options(thff_paper_fft PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Same adapter with the cuFFTDx comparison path (--library cufftdx). Built only
# when MATHDX_ROOT is set; cuFFTDx needs one compile-time SM, so one arch only.
# THFF_CUFFTDX_ENABLED records the outcome for artifact/table3/run.sh, so a build
# directory configured once with MATHDX_ROOT keeps measuring cuFFTDx.
set(THFF_CUFFTDX_ENABLED OFF CACHE INTERNAL "thff_paper_fft_cufftdx is configured")
if(MATHDX_ROOT)
  if(NOT EXISTS "${MATHDX_ROOT}/include/cufftdx.hpp")
    message(FATAL_ERROR "MATHDX_ROOT=${MATHDX_ROOT} does not contain include/cufftdx.hpp")
  endif()
  list(LENGTH THUNDERFFT_ARCHS thff_arch_count)
  if(thff_arch_count EQUAL 1)
    string(REGEX MATCH "^[0-9]+" thff_cufftdx_arch "${THUNDERFFT_ARCHS}")
    add_executable(thff_paper_fft_cufftdx paper_fft.cu)
    target_include_directories(thff_paper_fft_cufftdx PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/util
      ${CMAKE_CURRENT_SOURCE_DIR}/baseline
      ${MATHDX_ROOT}/include
      ${MATHDX_ROOT}/external/cutlass/include
    )
    target_compile_definitions(thff_paper_fft_cufftdx PRIVATE
      THFF_WITH_CUFFTDX=1 THFF_CUFFTDX_SM=${thff_cufftdx_arch}0)
    target_link_libraries(thff_paper_fft_cufftdx PRIVATE thunderfft CUDA::cudart CUDA::cufft)
    target_compile_options(thff_paper_fft_cufftdx PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
      $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
    set(THFF_CUFFTDX_ENABLED ON CACHE INTERNAL "thff_paper_fft_cufftdx is configured")
  else()
    message(STATUS "thff_paper_fft_cufftdx skipped: THUNDERFFT_ARCHS must name one architecture")
  endif()
endif()

# Table5 precision/range benchmark (separated source, same build environment)
add_executable(benchmark_table5_precision
  ${PROJECT_SOURCE_DIR}/artifact/table5/main.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/baseline/baseline_cufft.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/util/stat.cu
)

target_include_directories(benchmark_table5_precision PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/util
  ${CMAKE_CURRENT_SOURCE_DIR}/baseline
)

target_link_libraries(benchmark_table5_precision PRIVATE
  thunderfft
  CUDA::cudart
  CUDA::cufft
)

target_compile_options(benchmark_table5_precision PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Table 5 cuFFTDx FP16/FP32 rows (--precision cufftdx_fp16|cufftdx_fp32); no fast-math.
if(THFF_CUFFTDX_ENABLED)
  add_executable(benchmark_table5_precision_cufftdx
    ${PROJECT_SOURCE_DIR}/artifact/table5/main.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/baseline/baseline_cufft.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/util/stat.cu
  )
  target_include_directories(benchmark_table5_precision_cufftdx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/util
    ${CMAKE_CURRENT_SOURCE_DIR}/baseline
    ${MATHDX_ROOT}/include
    ${MATHDX_ROOT}/external/cutlass/include
  )
  target_compile_definitions(benchmark_table5_precision_cufftdx PRIVATE
    THFF_WITH_CUFFTDX=1 THFF_CUFFTDX_SM=${thff_cufftdx_arch}0)
  target_link_libraries(benchmark_table5_precision_cufftdx PRIVATE thunderfft CUDA::cudart CUDA::cufft)
  target_compile_options(benchmark_table5_precision_cufftdx PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )
endif()
