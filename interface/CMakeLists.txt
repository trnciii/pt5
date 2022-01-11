set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/pt5/core)

set(PYBIND11_PYTHON_VERSION 3.9)
add_subdirectory(ext/pybind11)

include_directories(ext/)

pybind11_add_module(core
	interface.cpp
	material.cpp
	scene.cpp
	util.hpp
)

target_link_libraries(core PRIVATE pt5)
