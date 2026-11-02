# Set MATHDX_ROOT here
set(MATHDX_ROOT "" CACHE PATH "Path to NVIDIA MathDX installation (e.g. /opt/nvidia/mathdx/25.06)")

if(NOT MATHDX_ROOT AND DEFINED ENV{MATHDX_ROOT})
  set(MATHDX_ROOT "$ENV{MATHDX_ROOT}")
endif()

add_subdirectory(conv2d)
add_subdirectory(conv1d)
# add_subdirectory(fft1d)
add_subdirectory(bench)
