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
