set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/pypt5)


set(PYBIND11_PYTHON_VERSION 3.9)
add_subdirectory(pybind11)


pybind11_add_module(pypt5 interface.cpp)

target_link_libraries(pypt5
	PRIVATE
	pt5
	cuda
	${CUDA_LIBRARIES}
	)
