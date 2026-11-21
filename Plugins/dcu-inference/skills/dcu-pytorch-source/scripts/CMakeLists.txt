cmake_minimum_required(VERSION 3.16.3)
project(torch_optest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

# ── LibTorch 路径（部署的独立运行时） ──
set(TORCH_ROOT    ./torch2.1.0)
set(TORCH_INCLUDE ${TORCH_ROOT}/include
                  ${TORCH_ROOT}/include/torch/csrc/api/include)
set(TORCH_LIBDIR  ${TORCH_ROOT}/lib)

# ── HIP/ROCm 路径 ──
set(HIP_ROOT      /opt/dtk-26.04)
set(HIP_INCLUDE   ${HIP_ROOT}/include)
set(HIP_LIBDIR    ${HIP_ROOT}/lib)

include_directories(${TORCH_INCLUDE} ${HIP_INCLUDE})
link_directories(${TORCH_LIBDIR} ${HIP_LIBDIR})

set(TORCH_LIBS torch torch_cpu torch_hip c10 c10_hip pthread dl rt)

# ── Target 1: Smoke Test ──
add_executable(demo demo.cpp)
target_link_libraries(demo ${TORCH_LIBS})
set_target_properties(demo PROPERTIES
    INSTALL_RPATH "${HIP_LIBDIR}"
    BUILD_RPATH   "${HIP_LIBDIR}")

# ── Target 2: 全量算子测试 ──
add_executable(op_test op_test.cpp)
target_link_libraries(op_test ${TORCH_LIBS})
set_target_properties(op_test PROPERTIES
    INSTALL_RPATH "${HIP_LIBDIR}"
    BUILD_RPATH   "${HIP_LIBDIR}")
