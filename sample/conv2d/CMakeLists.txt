add_executable(thff_conv
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/thunder.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cufftdx.cu
)

target_link_libraries(thff_conv PRIVATE thunderfft CUDA::cudart CUDA::cufft)
# cuFFTDx is compiled for one SM: the first THUNDERFFT_ARCHS entry (e.g. 80 -> 800).
list(GET THUNDERFFT_ARCHS 0 thff_conv_arch)
string(REGEX MATCH "^[0-9]+" thff_conv_arch "${thff_conv_arch}")
target_compile_definitions(thff_conv PRIVATE THFF_CONV_SM=${thff_conv_arch}0)

target_include_directories(thff_conv PRIVATE
  ${MATHDX_ROOT}/external/cutlass/include
  ${MATHDX_ROOT}/include
)

target_compile_options(thff_conv PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)


# Figure 5's cuDNN baseline, when cuDNN is installed (same CUDNN_LIB as sample/conv1d).
find_library(CUDNN_LIB cudnn)
if(CUDNN_LIB)
  # Headers next to the selected library, so -DCUDNN_LIB=... pins one cuDNN version.
  get_filename_component(thff_cudnn_dir "${CUDNN_LIB}" DIRECTORY)
  find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS "${thff_cudnn_dir}/../include" NO_DEFAULT_PATH)
  find_path(CUDNN_INCLUDE_DIR cudnn.h)
endif()
if(CUDNN_LIB)
  target_link_libraries(thff_conv PRIVATE ${CUDNN_LIB})
  target_include_directories(thff_conv BEFORE PRIVATE ${CUDNN_INCLUDE_DIR})
  # cuDNN's directory may also hold another CUDA's cudart/cuFFT: search this toolkit first.
  set_target_properties(thff_conv PROPERTIES BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}")
  target_compile_definitions(thff_conv PRIVATE THFF_WITH_CUDNN=1)
endif()
