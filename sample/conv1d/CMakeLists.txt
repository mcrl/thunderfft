add_executable(thff_conv1d
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
)

set(THFF_CONV1D_TILES_PER_WARP 1 CACHE STRING "Logical FFT blocks handled per warp (compile-time constant)")
target_compile_definitions(thff_conv1d PRIVATE
  THFF_CONV1D_TILES_PER_WARP=${THFF_CONV1D_TILES_PER_WARP}
)

target_link_libraries(thff_conv1d PRIVATE thunderfft CUDA::cudart CUDA::cufft)

target_compile_options(thff_conv1d PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

add_executable(thff_conv1d_fullfft4096
  ${CMAKE_CURRENT_SOURCE_DIR}/conv1d_fullfft_l4096_thunder.cu
)
set(THFF_LMID256_PAD_PERIOD 64 CACHE STRING "Mid-roundtrip layout pad_period for N=256")
set(THFF_LMID256_PAD 1 CACHE STRING "Mid-roundtrip layout pad for N=256")
set(THFF_LMID1024_PAD_PERIOD 16 CACHE STRING "Mid-roundtrip layout pad_period for N=1024")
set(THFF_LMID1024_PAD 1 CACHE STRING "Mid-roundtrip layout pad for N=1024")
set(THFF_SKIP_MID_ROUNDTRIP 0 CACHE STRING "Skip reg2smem->smem2reg in N!=4096 path (perf-only)")
target_compile_definitions(thff_conv1d_fullfft4096 PRIVATE
  THFF_LMID256_PAD_PERIOD=${THFF_LMID256_PAD_PERIOD}
  THFF_LMID256_PAD=${THFF_LMID256_PAD}
  THFF_LMID1024_PAD_PERIOD=${THFF_LMID1024_PAD_PERIOD}
  THFF_LMID1024_PAD=${THFF_LMID1024_PAD}
  THFF_SKIP_MID_ROUNDTRIP=${THFF_SKIP_MID_ROUNDTRIP}
)
target_link_libraries(thff_conv1d_fullfft4096 PRIVATE thunderfft CUDA::cudart CUDA::cufft)
target_compile_options(thff_conv1d_fullfft4096 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

find_library(CUDNN_LIB cudnn)
if(CUDNN_LIB)
  # Headers next to the selected library, so -DCUDNN_LIB=... pins one cuDNN version.
  get_filename_component(thff_cudnn_dir "${CUDNN_LIB}" DIRECTORY)
  find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS "${thff_cudnn_dir}/../include" NO_DEFAULT_PATH)
  find_path(CUDNN_INCLUDE_DIR cudnn.h)
endif()
if(CUDNN_LIB)
  add_executable(thff_conv1d_compare
    ${CMAKE_CURRENT_SOURCE_DIR}/conv1d_compare_cudnn_thunder.cu
  )
  target_link_libraries(thff_conv1d_compare PRIVATE CUDA::cudart ${CUDNN_LIB})
  target_include_directories(thff_conv1d_compare BEFORE PRIVATE ${CUDNN_INCLUDE_DIR})
  # cuDNN's directory may also hold another CUDA's cudart/cuFFT: search this toolkit first.
  set_target_properties(thff_conv1d_compare PROPERTIES BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}")
  target_compile_options(thff_conv1d_compare PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )
else()
  message(WARNING "cudnn library not found; thff_conv1d_compare target will not be built")
endif()
