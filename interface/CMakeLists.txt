set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/pt5/core)

add_subdirectory(ext/pybind11)

include_directories(ext/)

pybind11_add_module(core
	interface.cpp
	material.cpp
	scene.cpp
	util.hpp
)

target_link_libraries(core PRIVATE pt5)
