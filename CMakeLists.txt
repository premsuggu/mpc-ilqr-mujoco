cmake_minimum_required(VERSION 3.16)
project(humanoid_mpc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Suppress warnings from external libraries - platform specific
if(MSVC)
    # Windows: Suppress all warnings with /w
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w")
else()
    # Linux/macOS: Suppress warnings with -w
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
endif()

# Platform-specific setup for conda environment
if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_PREFIX_PATH}")
    
    if(WIN32)
        # Windows: conda packages are in Library subdirectory
        include_directories("$ENV{CONDA_PREFIX}/Library/include")
        include_directories("$ENV{CONDA_PREFIX}/Library/include/eigen3")
        link_directories("$ENV{CONDA_PREFIX}/Library/lib")
        set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/Library;${CMAKE_PREFIX_PATH}")
    else()
        # Linux/macOS: standard paths
        include_directories("$ENV{CONDA_PREFIX}/include")
        include_directories("$ENV{CONDA_PREFIX}/include/eigen3")
        link_directories("$ENV{CONDA_PREFIX}/lib")
    endif()
endif()

# Include project headers
include_directories(include)

# Find required packages (let CMake locate them)
find_package(pinocchio QUIET)
find_package(casadi QUIET)
find_package(yaml-cpp QUIET)

# Main executable only
add_executable(humanoid_mpc
    src/robot_utils.cpp
    src/ilqr.cpp
    src/mpc.cpp
    src/derivatives.cpp
    src/config.cpp
    app/humanoid_mpc.cpp
)

# Explicitly set include directories for all dependencies (critical for Windows)
if(pinocchio_FOUND)
    target_include_directories(humanoid_mpc PRIVATE ${pinocchio_INCLUDE_DIRS})
    message(STATUS "Pinocchio include dirs: ${pinocchio_INCLUDE_DIRS}")
endif()

if(casadi_FOUND)
    target_include_directories(humanoid_mpc PRIVATE ${casadi_INCLUDE_DIRS})
    message(STATUS "CasADi include dirs: ${casadi_INCLUDE_DIRS}")
endif()

if(yaml-cpp_FOUND)
    target_include_directories(humanoid_mpc PRIVATE ${yaml-cpp_INCLUDE_DIRS})
    message(STATUS "yaml-cpp include dirs: ${yaml-cpp_INCLUDE_DIRS}")
endif()

# Find OpenGL (optional for headless CI environments)
find_package(OpenGL)
if(OPENGL_FOUND)
    set(OPENGL_LIBS ${OPENGL_LIBRARIES})
else()
    message(WARNING "OpenGL not found - visualization may not work")
    set(OPENGL_LIBS "")
endif()

# Platform-specific libraries
if(UNIX AND NOT APPLE)
    # Linux
    set(PLATFORM_LIBS pthread dl)
elseif(APPLE)
    # macOS
    set(PLATFORM_LIBS pthread dl)
elseif(WIN32)
    # Windows
    set(PLATFORM_LIBS "")
endif()

# Link libraries
target_link_libraries(humanoid_mpc 
    mujoco 
    pinocchio_default 
    pinocchio_parsers 
    pinocchio_casadi 
    casadi
    glfw
    ${OPENGL_LIBS}
    ${PLATFORM_LIBS}
    yaml-cpp
)

# Compiler flags - platform specific
if(MSVC)
    # Windows: Use MSVC optimization flags
    target_compile_options(humanoid_mpc PRIVATE /O2 /W0)
else()
    # Linux/macOS: Use GCC/Clang flags
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
