cmake_minimum_required(VERSION 3.10)
project(seePlusPlus)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES "src/*.cpp")

add_executable(seePlusPlus ${SOURCES})

target_link_libraries(seePlusPlus ${OpenCV_LIBS})

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Set compile options for Debug and Release
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug build")
    target_compile_definitions(seePlusPlus PRIVATE DEBUG_MODE=1)
    target_compile_options(seePlusPlus PRIVATE -g -O0)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Release build")
    target_compile_definitions(seePlusPlus PRIVATE NDEBUG)
    target_compile_options(seePlusPlus PRIVATE -O3)
endif()