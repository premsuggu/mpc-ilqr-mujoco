cmake_minimum_required(VERSION 3.20)
project(humanoid_mpc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Building Humanoid MPC...")
message(STATUS "Conda environment: $ENV{CONDA_PREFIX}")

# Use conda environment for finding packages
if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX};$ENV{CONDA_PREFIX}/Library;${CMAKE_PREFIX_PATH}")
    message(STATUS "Using conda prefix: $ENV{CONDA_PREFIX}")
endif()

# Cross-platform package finding
find_package(PkgConfig QUIET)

# Find MuJoCo (conda package: mujoco)
find_package(mujoco QUIET)
if(mujoco_FOUND)
    message(STATUS "✅ Found MuJoCo: ${mujoco_VERSION}")
else()
    message(STATUS "Looking for MuJoCo in conda environment...")
    if(DEFINED ENV{CONDA_PREFIX})
        # Try to find mujoco manually in conda environment
        find_path(MUJOCO_INCLUDE_DIR 
            NAMES mujoco/mujoco.h
            PATHS "$ENV{CONDA_PREFIX}/include" "$ENV{CONDA_PREFIX}/Library/include"
            NO_DEFAULT_PATH)
        find_library(MUJOCO_LIBRARY 
            NAMES mujoco
            PATHS "$ENV{CONDA_PREFIX}/lib" "$ENV{CONDA_PREFIX}/Library/lib"
            NO_DEFAULT_PATH)
        
        if(MUJOCO_INCLUDE_DIR AND MUJOCO_LIBRARY)
            message(STATUS "✅ Found MuJoCo manually: ${MUJOCO_LIBRARY}")
            set(MUJOCO_FOUND TRUE)
        endif()
    endif()
endif()

# Find Eigen3 (conda package: eigen)
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    message(STATUS "✅ Found Eigen3: ${Eigen3_VERSION}")
else()
    # Try to find eigen manually in conda environment
    if(DEFINED ENV{CONDA_PREFIX})
        find_path(EIGEN3_INCLUDE_DIR 
            NAMES Eigen/Dense
            PATHS "$ENV{CONDA_PREFIX}/include/eigen3" "$ENV{CONDA_PREFIX}/Library/include/eigen3"
                  "$ENV{CONDA_PREFIX}/include" "$ENV{CONDA_PREFIX}/Library/include"
            NO_DEFAULT_PATH)
        
        if(EIGEN3_INCLUDE_DIR)
            message(STATUS "✅ Found Eigen3 manually: ${EIGEN3_INCLUDE_DIR}")
            set(EIGEN3_FOUND TRUE)
        endif()
    endif()
endif()

# Find GLFW (conda package: glfw)
find_package(glfw3 QUIET)
if(glfw3_FOUND)
    message(STATUS "✅ Found GLFW3: ${glfw3_VERSION}")
else()
    # Try to find glfw manually
    if(DEFINED ENV{CONDA_PREFIX})
        find_library(GLFW3_LIBRARY 
            NAMES glfw3 glfw
            PATHS "$ENV{CONDA_PREFIX}/lib" "$ENV{CONDA_PREFIX}/Library/lib"
            NO_DEFAULT_PATH)
        find_path(GLFW3_INCLUDE_DIR 
            NAMES GLFW/glfw3.h
            PATHS "$ENV{CONDA_PREFIX}/include" "$ENV{CONDA_PREFIX}/Library/include"
            NO_DEFAULT_PATH)
            
        if(GLFW3_LIBRARY AND GLFW3_INCLUDE_DIR)
            message(STATUS "✅ Found GLFW3 manually: ${GLFW3_LIBRARY}")
            set(GLFW3_FOUND TRUE)
        endif()
    endif()
endif()

# Include directories
include_directories(include)

# Add conda include paths
if(DEFINED ENV{CONDA_PREFIX})
    include_directories("$ENV{CONDA_PREFIX}/include")
    if(WIN32)
        include_directories("$ENV{CONDA_PREFIX}/Library/include")
    endif()
endif()

# Add specific include directories if found
if(MUJOCO_INCLUDE_DIR)
    include_directories(${MUJOCO_INCLUDE_DIR})
endif()
if(EIGEN3_INCLUDE_DIR)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()
if(GLFW3_INCLUDE_DIR)
    include_directories(${GLFW3_INCLUDE_DIR})
endif()

# Source files
set(SOURCES
    src/robot_utils.cpp
    src/ilqr.cpp
    src/mpc.cpp
    app/humanoid_mpc.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
if(mujoco_FOUND)
    target_link_libraries(${PROJECT_NAME} mujoco::mujoco)
elseif(MUJOCO_LIBRARY)
    target_link_libraries(${PROJECT_NAME} ${MUJOCO_LIBRARY})
endif()

if(Eigen3_FOUND)
    target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
endif()

if(glfw3_FOUND)
    target_link_libraries(${PROJECT_NAME} glfw)
elseif(GLFW3_LIBRARY)
    target_link_libraries(${PROJECT_NAME} ${GLFW3_LIBRARY})
endif()

# Platform-specific linking
if(WIN32)
    target_link_libraries(${PROJECT_NAME} opengl32 user32 gdi32 shell32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
elseif(APPLE)
    find_library(OPENGL_LIBRARY OpenGL)
    find_library(COCOA_LIBRARY Cocoa)
    find_library(IOKIT_LIBRARY IOKit)
    find_library(COREVIDEO_LIBRARY CoreVideo)
    target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARY} ${COCOA_LIBRARY} ${IOKIT_LIBRARY} ${COREVIDEO_LIBRARY})
else()
    # Linux
    find_package(OpenGL QUIET)
    if(OPENGL_FOUND)
        target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES} pthread dl)
    else()
        # Fallback for systems where OpenGL detection is problematic
        find_library(GL_LIBRARY NAMES GL OpenGL PATHS /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu)
        if(GL_LIBRARY)
            target_link_libraries(${PROJECT_NAME} ${GL_LIBRARY} pthread dl)
        else()
            message(WARNING "OpenGL not found, linking with -lGL")
            target_link_libraries(${PROJECT_NAME} GL pthread dl)
        endif()
    endif()
endif()

# Compiler options
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W3)
    target_compile_definitions(${PROJECT_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
endif()

# Print build summary
message(STATUS "")
message(STATUS "🤖 ===== BUILD CONFIGURATION SUMMARY =====")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Conda prefix: $ENV{CONDA_PREFIX}")
if(mujoco_FOUND OR MUJOCO_FOUND)
    message(STATUS "✅ MuJoCo: FOUND")
else()
    message(STATUS "❌ MuJoCo: NOT FOUND")
endif()
if(Eigen3_FOUND OR EIGEN3_FOUND)
    message(STATUS "✅ Eigen3: FOUND") 
else()
    message(STATUS "❌ Eigen3: NOT FOUND")
endif()
if(glfw3_FOUND OR GLFW3_FOUND)
    message(STATUS "✅ GLFW3: FOUND")
else()
    message(STATUS "❌ GLFW3: NOT FOUND")
endif()
message(STATUS "==========================================")
message(STATUS "")
