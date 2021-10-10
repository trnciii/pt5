
if( POLICY CMP0072 )
  cmake_policy(SET CMP0072 NEW)
endif()

find_package(OpenGL REQUIRED)

find_package(glfw3 3.3 REQUIRED)


include_directories(${PROJECT_SOURCE_DIR}/libpt5)

add_executable(main main.cpp)
target_link_libraries(main
	pt5

	cuda
	${CUDA_LIBRARIES}

	glfw
	${OPENGL_LIBRARIES}
	)