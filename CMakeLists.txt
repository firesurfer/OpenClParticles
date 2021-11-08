cmake_minimum_required(VERSION 3.5)

project(fluids)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
endif(CCACHE_FOUND)

set(CMAKE_CXX_STANDARD 20)

add_compile_options(-Wall -Wextra  -O3 -march=native -g -fopenmp) #-
add_link_options(-fopenmp)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Find installed Open3D, which exports Open3D::Open3D
find_package(Open3D REQUIRED)
find_package(OpenCL REQUIRED)

include_directories(
    ./src
    ./ext/
    ./ext/compute/include
)


add_custom_target(copy_kernels
    COMMAND bash -c "mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/kernels "
    COMMAND bash -c "cp ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/* ${CMAKE_CURRENT_BINARY_DIR}/kernels"

)


add_executable(fluids_cpu
    src/main.cpp
    src/ParticleSimulation.h
)

target_link_libraries(fluids_cpu
    Threads::Threads
    Open3D::Open3D
)


add_executable(fluids_gpu
    src/main_gpu.cpp
    src/ParticleSimulation.h
)
add_dependencies(fluids_gpu copy_kernels)

target_link_libraries(fluids_gpu
    Threads::Threads
    Open3D::Open3D
    OpenCL::OpenCL
)

