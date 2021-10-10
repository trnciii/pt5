
add_executable(main main.cpp)
target_link_libraries(main
	pt5

	cuda
	${CUDA_LIBRARIES}

	glfw
	${OPENGL_LIBRARIES}
	)