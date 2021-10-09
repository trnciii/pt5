cmake_minimum_required(VERSION 3.4)
project(pt5)


set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/cmake"
  ${CMAKE_MODULE_PATH}
  )


SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


set(CMAKE_CXX_STANDARD 17)


# cuda and optix
find_package(CUDA REQUIRED)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)


set(OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR} CACHE PATH "Path to OptiX installed location.")
find_package(OptiX REQUIRED)
include_directories(${OptiX_INCLUDE})


# add
add_subdirectory(libpt5)
add_subdirectory(host)
add_subdirectory(interface)
