# macros
find_program(BIN2C bin2c DOC "Path to the cuda-sdk bin2c executable.")

macro(cuda_compile_and_embed output_var cuda_file)
	set(c_var_name ${output_var})
	cuda_compile_ptx(ptx_files ${cuda_file} OPTIONS --generate-line-info -use_fast_math --keep)
	list(GET ptx_files 0 ptx_file)
 	set(embedded_file ${ptx_file}_embedded.c)
	message("adding rule to compile and embed ${cuda_file} to \"const char ${var_name}[];\"")
	add_custom_command(
    	OUTPUT ${embedded_file}
    	COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
    	DEPENDS ${ptx_file}
    	COMMENT "compiling (and embedding ptx from) ${cuda_file}"
    )
	set(${output_var} ${embedded_file})
endmacro()


# gl
if( POLICY CMP0072 )
  cmake_policy(SET CMP0072 NEW)
endif()

find_package(OpenGL REQUIRED)


set(CMAKE_POSITION_INDEPENDENT_CODE ON) # kind of fPIC

cuda_compile_and_embed(embedded_ptx_code
	device/device.cu
	device/intersection.cuh
	device/deviceutil.cuh
	device/math.cuh
	)

add_library(pt5
	${embedded_ptx_code}
	camera.hpp
	CUDABuffer.hpp
	LaunchParams.hpp
	material.h
	mesh.hpp
	optix.hpp
	scene.cpp
	scene.hpp
	texture.hpp
	tracer.hpp
	tracer.cpp
	vector_math.h
	view.hpp
	view.cpp
	)

target_link_libraries(pt5
	cuda
	${CUDA_LIBRARIES}

	${OPENGL_LIBRARIES}
	)