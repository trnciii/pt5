find_package(glfw3 3.3 REQUIRED)
add_executable(main main.cpp)
target_link_libraries(main pt5 glfw)