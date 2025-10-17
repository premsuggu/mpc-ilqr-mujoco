cmake_minimum_required(VERSION 3.16)
project(humanoid_mpc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Suppress/adjust warnings by compiler
if(MSVC)
    # MSVC: disable warnings, optimize, allow big object files, and parallel compilation
    add_compile_options(/W0 /O2 /bigobj /MP)
else()
    # GCC/Clang
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
endif()

# Find packages in conda environment
if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_PREFIX_PATH}")
    # Typical Unix-like layout
    include_directories("$ENV{CONDA_PREFIX}/include")
    include_directories("$ENV{CONDA_PREFIX}/include/eigen3")
    link_directories("$ENV{CONDA_PREFIX}/lib")
    # Conda on Windows uses Library/include and Library/lib
    if(WIN32)
        include_directories("$ENV{CONDA_PREFIX}/Library/include")
        include_directories("$ENV{CONDA_PREFIX}/Library/include/eigen3")
        link_directories("$ENV{CONDA_PREFIX}/Library/lib")
        add_definitions(-DNOMINMAX)
    endif()
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

# Fix Pinocchio hidden symbol issue on Linux CI
if(UNIX AND NOT APPLE)
    # Force default visibility for all symbols (fixes hidden symbol errors)
    target_compile_options(humanoid_mpc PRIVATE -fvisibility=default)
    # Force linker to include all dependencies
    target_link_options(humanoid_mpc PRIVATE -Wl,--no-as-needed)
endif()

# Link libraries
# GLFW library name differs across platforms in Conda (glfw on Unix, glfw3dll on Windows)
set(GLFW_LIB_NAME glfw)
if(WIN32)
    # Conda-forge provides glfw3dll.lib on Windows
    set(GLFW_LIB_NAME glfw3dll)
endif()

target_link_libraries(humanoid_mpc
    mujoco
    pinocchio_default
    pinocchio_parsers
    pinocchio_casadi
    casadi
    yaml-cpp
    ${GLFW_LIB_NAME}
)

if(UNIX AND NOT APPLE)
    # Linux-specific system libs
    target_link_libraries(humanoid_mpc GL pthread dl)
    # Explicitly link Pinocchio transitive dependencies (fixes hidden symbol errors)
    target_link_libraries(humanoid_mpc 
        urdfdom_model
        urdfdom_world
        console_bridge
    )
elseif(WIN32)
    # Windows system libs typically required by OpenGL/GLFW
    target_link_libraries(humanoid_mpc opengl32 user32 gdi32 shell32)
endif()

# Compiler flags for main executable only
if(NOT MSVC)
    target_compile_options(humanoid_mpc PRIVATE -O3)
endif()

# Optional profiling flag (OFF by default for zero overhead)
option(ENABLE_PROFILING "Enable performance profiling" OFF)
if(ENABLE_PROFILING)
    target_compile_definitions(humanoid_mpc PRIVATE ENABLE_PROFILING)
    message(STATUS "Performance profiling: ENABLED")
else()
    message(STATUS "Performance profiling: DISABLED (use -DENABLE_PROFILING=ON to enable)")
endif()
