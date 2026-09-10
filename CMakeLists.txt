cmake_minimum_required(VERSION 3.18)
project(bitplane_search LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(bitplane_index cpp/index.cpp cpp/key_index.cpp cpp/bindings.cpp)
target_compile_features(bitplane_index PRIVATE cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(bitplane_index PRIVATE -Wall -Wextra -Wpedantic)
endif()
install(TARGETS bitplane_index LIBRARY DESTINATION .)
