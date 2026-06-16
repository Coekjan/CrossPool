cmake_minimum_required(VERSION 3.26)

project(xpool LANGUAGES CXX)

add_library(xpool_abi INTERFACE)
target_include_directories(xpool_abi INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/cext/include)
target_compile_features(xpool_abi INTERFACE cxx_std_20)
