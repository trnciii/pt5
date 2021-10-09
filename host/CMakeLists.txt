include_directories(${PROJECT_SOURCE_DIR}/libpt5)

add_executable(main main.cpp)

target_link_libraries(main
	pt5
	cuda
	${CUDA_LIBRARIES}
	)