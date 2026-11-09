add_executable(thff_conv
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/thunder_tiled64.cu
)

target_link_libraries(thff_conv PRIVATE thunderfft CUDA::cudart CUDA::cufft)
# cuFFT and ThunderFFT are available without MathDX.
if(MATHDX_ROOT AND EXISTS "${MATHDX_ROOT}/include/cufftdx.hpp")
  target_sources(thff_conv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cufftdx.cu)
  list(GET THUNDERFFT_ARCHS 0 thff_conv_arch)
  string(REGEX MATCH "^[0-9]+" thff_conv_arch "${thff_conv_arch}")
  target_compile_definitions(thff_conv PRIVATE THFF_WITH_CUFFTDX=1 THFF_CONV_SM=${thff_conv_arch}0)
  target_include_directories(thff_conv PRIVATE ${MATHDX_ROOT}/external/cutlass/include ${MATHDX_ROOT}/include)
  set(THFF_CONV_CUFFTDX_ENABLED ON CACHE INTERNAL "Figure 4 cuFFTDx backend" FORCE)
else()
  set(THFF_CONV_CUFFTDX_ENABLED OFF CACHE INTERNAL "Figure 4 cuFFTDx backend" FORCE)
endif()

target_compile_options(thff_conv PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)


# Figure 4's cuDNN baseline, when cuDNN is installed (same CUDNN_LIB as sample/conv1d).
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

# Optional: OpenMP parallelizes the sampled CPU reference check over all cores.
# Without it the check runs on one thread with the same result.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(thff_conv PRIVATE OpenMP::OpenMP_CXX)
  target_compile_options(thff_conv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
endif()
