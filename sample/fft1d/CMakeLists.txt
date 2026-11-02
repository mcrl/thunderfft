add_executable(thff_fft
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
)

# Link header-only library (exports include dirs + cudart)
target_link_libraries(thff_fft PRIVATE thunderfft CUDA::cudart CUDA::cufft)

# Optional: common flags
target_compile_options(thff_fft PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)