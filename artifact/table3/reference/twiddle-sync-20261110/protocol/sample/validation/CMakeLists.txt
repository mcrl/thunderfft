add_executable(thff_validate fft_validation.cu)
target_link_libraries(thff_validate PRIVATE thunderfft CUDA::cudart CUDA::cufft)
target_compile_options(thff_validate PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# This separate translation unit contains no FFT/MMA code. CUDA 12 can therefore
# validate per-device cache ownership on Volta as well as Tensor Core FFT GPUs.
add_executable(thff_validate_cache cache_validation.cu)
target_link_libraries(thff_validate_cache PRIVATE thunderfft CUDA::cudart)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13)
  set_property(TARGET thff_validate_cache PROPERTY CUDA_ARCHITECTURES "70;80")
  target_compile_definitions(thff_validate_cache PRIVATE THUNDERFFT_CACHE_MIN_CC=70)
else()
  set_property(TARGET thff_validate_cache PROPERTY CUDA_ARCHITECTURES "80")
  target_compile_definitions(thff_validate_cache PRIVATE THUNDERFFT_CACHE_MIN_CC=80)
endif()
