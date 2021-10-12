set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/interface/pt5)

set(PYBIND11_PYTHON_VERSION 3.9)
add_subdirectory(pybind11)


pybind11_add_module(core interface.cpp)

target_link_libraries(core PRIVATE pt5 glfw)
