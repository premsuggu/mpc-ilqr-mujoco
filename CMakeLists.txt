cmake_minimum_required(VERSION 3.16)
project(humanoid_mpc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Suppress warnings from external libraries
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

# Find packages in conda environment
if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_PREFIX_PATH}")
    include_directories("$ENV{CONDA_PREFIX}/include")
    include_directories("$ENV{CONDA_PREFIX}/include/eigen3")
    link_directories("$ENV{CONDA_PREFIX}/lib")
endif()

# Include directories
include_directories(include)

# Main executable only
add_executable(humanoid_mpc
    src/robot_utils.cpp
    src/ilqr.cpp
    src/mpc.cpp
    src/derivatives.cpp
    src/config.cpp
    app/humanoid_mpc.cpp
)

# Link libraries
target_link_libraries(humanoid_mpc 
    mujoco 
    pinocchio_default 
    pinocchio_parsers 
    pinocchio_casadi 
    casadi
    glfw
    GL pthread dl
    yaml-cpp
)

# Compiler flags for main executable only
target_compile_options(humanoid_mpc PRIVATE -O3)

# Optional profiling flag (OFF by default for zero overhead)
option(ENABLE_PROFILING "Enable performance profiling" OFF)
if(ENABLE_PROFILING)
    target_compile_definitions(humanoid_mpc PRIVATE ENABLE_PROFILING)
    message(STATUS "Performance profiling: ENABLED")
else()
    message(STATUS "Performance profiling: DISABLED (use -DENABLE_PROFILING=ON to enable)")
endif()
