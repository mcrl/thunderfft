add_executable(thff_conv
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/thunder.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cufftdx.cu
)

target_link_libraries(thff_conv PRIVATE thunderfft CUDA::cudart CUDA::cufft)

target_include_directories(thff_conv PRIVATE
  ${MATHDX_ROOT}/external/cutlass/include
  ${MATHDX_ROOT}/include
)

target_compile_options(thff_conv PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

