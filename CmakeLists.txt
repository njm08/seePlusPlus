cmake_minimum_required(VERSION 3.10)
project(seePlusPlus)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES "src/*.cpp")

add_executable(seePlusPlus ${SOURCES})

target_link_libraries(seePlusPlus ${OpenCV_LIBS})

# Debug mode options
option(DEBUG_MODE "Enable debug mode" ON)

if(DEBUG_MODE)
    message(STATUS "Debug mode enabled")
    target_compile_definitions(seePlusPlus PRIVATE DEBUG_MODE=1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
endif()