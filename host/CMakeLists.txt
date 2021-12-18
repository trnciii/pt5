find_package(glfw3 3.3 REQUIRED)
find_package(TBB REQUIRED tbb)
add_executable(main main.cpp)
target_link_libraries(main pt5 glfw ${TBB_IMPORTED_TARGETS})