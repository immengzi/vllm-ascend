cmake_minimum_required(VERSION 3.16)
project(vllm_ascend_C)

# include(CheckCXXcompilerFlag)
# check_cxx_compiler_flag("-std=c++17", COMPILER_SUPPORTS_CXX17)
set(CMAKE_CXX_STANDARD 17)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${VLLM_PYTHON_PATH}")

# TODO: Add 3.12 back when torch-npu support 3.12
set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11")

find_package(pybind11 REQUIRED)

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
set(VLLM_ASCEND_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")

find_package(Torch REQUIRED)

run_python(TORCH_VERSION
  "import torch; print(torch.__version__)" "Failed to locate torch path")
# check torch version is 2.9.0
if(NOT ${TORCH_VERSION} VERSION_EQUAL "2.9.0")
  message(FATAL_ERROR "Expected PyTorch version 2.9.0, but found ${TORCH_VERSION}")
endif()

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION ${SOC_VERSION})
message(STATUS "Detected SOC version: ${SOC_VERSION}")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRINGS "Build type Release/Debug (default Release)" FORCE)
endif()

if (CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRINGS "path to install()")
endif()

set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})
if(EXISTS ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

file(GLOB KERNEL_FILES
${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/*.cpp)

set(VLLM_ASCEND_CUSTOM_OP
    ${KERNEL_FILES}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/batch_matmul_transpose/op_kernel/batch_matmul_transpose_kernel.cpp
)

set(VLLM_ASCEND_CUSTOM_OP_EXCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/bgmv_expand.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/bgmv_shrink.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/sgmv_expand.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/sgmv_shrink.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/batch_matmul_transpose/op_kernel/batch_matmul_transpose_kernel.cpp
)

if(SOC_VERSION MATCHES "ascend310p.*")
    message(STATUS "310P hardware detected: disabling MLAPO operators")
    message(STATUS "310P hardware detected: excluding batch_matmul_transpose operators")
    list(REMOVE_ITEM VLLM_ASCEND_CUSTOM_OP ${VLLM_ASCEND_CUSTOM_OP_EXCLUDE})
endif()

ascendc_library(vllm_ascend_kernels SHARED
    ${VLLM_ASCEND_CUSTOM_OP}
)

message("TORCH_NPU_PATH is ${TORCH_NPU_PATH}")

if(SOC_VERSION MATCHES "ascend310p.*")
    file(GLOB VLLM_ASCEND_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/aclnn_torch_adapter/*.cpp)
else()
    file(GLOB VLLM_ASCEND_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/aclnn_torch_adapter/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/batch_matmul_transpose/op_host/tiling/tiling_data.cpp)
endif()

include_directories(
  ${pybind11_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_PATH}
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_PATH}/include
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_HOME_PATH}/aarch64-linux/include/experiment/platform
  ${ASCEND_HOME_PATH}/x86_64-linux/include/experiment/platform
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/batch_matmul_transpose/op_host
)

set(
  INCLUDES
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_INCLUDE_DIRS}
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_HOME_PATH}/aarch64-linux/include/experiment/platform
  ${ASCEND_HOME_PATH}/x86_64-linux/include/experiment/platform
)

# ---------------------------------------------------------------------------
# Optional: SHMEM dynamic memory allocator module
#
# Enable with:   cmake -DENABLE_SHMEM=ON -DSHMEM_HOME=/path/to/shmem/install …
# Or at pip-install time set ENABLE_SHMEM=1 and SHMEM_HOME env vars.
#
# Why a separate pybind11_add_module?
#   • camem_allocator.cpp (compiled into vllm_ascend_C) already exports
#     my_malloc / my_free.  Adding a second definition to the same .so would
#     cause a linker duplicate-symbol error.
#   • libshmem.so is an optional dependency; gating it behind ENABLE_SHMEM
#     keeps the default build free of that requirement.
#   • NPUPluggableAllocator loads a .so by filesystem path and resolves
#     my_malloc / my_free via dlsym.  The same .so that serves as a Python
#     extension (PyInit_shmem_allocator) can also serve this role.
option(ENABLE_SHMEM "Build the SHMEM-backed NPU pluggable allocator" OFF)

if(ENABLE_SHMEM)
  set(SHMEM_HOME "$ENV{SHMEM_HOME}" CACHE PATH
      "Root of the installed SHMEM library (e.g. shmem/install/shmem)")
  if(NOT SHMEM_HOME)
    set(SHMEM_HOME "/usr/local/Ascend/shmem/latest")
    message(STATUS "SHMEM_HOME not set, using default: ${SHMEM_HOME}")
  else()
    message(STATUS "SHMEM_HOME: ${SHMEM_HOME}")
  endif()

  pybind11_add_module(shmem_allocator
      ${CMAKE_CURRENT_SOURCE_DIR}/csrc/shmem_allocator.cpp)

  target_include_directories(shmem_allocator PRIVATE
      ${SHMEM_HOME}/include
      ${pybind11_INCLUDE_DIRS}
      ${PYTHON_INCLUDE_PATH}
      ${TORCH_INCLUDE_DIRS}
      ${TORCH_NPU_PATH}/include
      ${ASCEND_HOME_PATH}/include)

  target_link_directories(shmem_allocator PRIVATE
      ${TORCH_LIBRARY_DIRS}
      ${TORCH_NPU_PATH}/lib/
      ${ASCEND_HOME_PATH}/lib64
      ${SHMEM_HOME}/lib)

  target_link_libraries(shmem_allocator PUBLIC
      ${TORCH_LIBRARIES}
      torch_npu
      ascendcl
      shmem)

  # Embed RPATH so the .so can find libshmem.so at runtime without LD_LIBRARY_PATH.
  set_target_properties(shmem_allocator PROPERTIES
      INSTALL_RPATH "$ORIGIN:${SHMEM_HOME}/lib"
      BUILD_WITH_INSTALL_RPATH TRUE)

  install(TARGETS shmem_allocator DESTINATION ${VLLM_ASCEND_INSTALL_PATH})
  message(STATUS "SHMEM allocator module enabled")
endif()
# ---------------------------------------------------------------------------

pybind11_add_module(vllm_ascend_C ${VLLM_ASCEND_SRC})

target_link_directories(
  vllm_ascend_C
  PRIVATE
  ${TORCH_LIBRARY_DIRS}
  ${TORCH_NPU_PATH}/lib/
  ${ASCEND_HOME_PATH}/lib64
)

target_link_libraries(
  vllm_ascend_C
  PUBLIC
  ${TORCH_LIBRARIES}
  torch_npu
  vllm_ascend_kernels
  ascendcl
  tiling_api
  register
  platform
  ascendalog
  dl
  opapi
)

target_link_options(vllm_ascend_C PRIVATE "-Wl,-rpath,$ORIGIN:$ORIGIN/lib:$ORIGIN/_cann_ops_custom/vendors/vllm-ascend/op_api/lib")

install(TARGETS vllm_ascend_C vllm_ascend_kernels DESTINATION ${VLLM_ASCEND_INSTALL_PATH})
