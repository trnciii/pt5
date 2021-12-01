#pragma once

#include "optix.hpp"
#include <vector>

namespace pt5{

struct CUDABuffer {
	inline CUdeviceptr d_pointer() const
	{ return (CUdeviceptr)d_ptr; }

	//! re-size buffer to given number of bytes
	void resize(size_t size, CUstream stream)
	{
		if (d_ptr) free(stream);
		alloc(size, stream);
	}

	//! allocate to given number of bytes
	void alloc(size_t size, CUstream stream)
	{
		assert(d_ptr == nullptr);
		this->sizeInBytes = size;
		CUDA_CHECK(cudaMallocAsync( (void**)&d_ptr, sizeInBytes, stream));
	}

	//! free allocated memory
	void free(CUstream stream)
	{
		CUDA_CHECK(cudaFreeAsync(d_ptr, stream));
		d_ptr = nullptr;
		sizeInBytes = 0;
	}

	template<typename T>
	void alloc_and_upload(const std::vector<T> &vt, CUstream stream)
	{
		alloc(vt.size()*sizeof(T), stream);
		upload((const T*)vt.data(),vt.size(), stream);
	}

	template<typename T>
	void alloc_and_upload(const T& t, CUstream stream){
		alloc(sizeof(T), stream);
		upload(&t, 1, stream);
	}

	template<typename T>
	void upload(const T *t, size_t count, CUstream stream)
	{
		assert(d_ptr != nullptr);
		assert(sizeInBytes == count*sizeof(T));
		CUDA_CHECK(cudaMemcpyAsync(d_ptr, (void *)t, count*sizeof(T), cudaMemcpyHostToDevice, stream));
	}

	template<typename T>
	void download(T *t, size_t count, CUstream stream)
	{
		assert(d_ptr != nullptr);
		assert(sizeInBytes == count*sizeof(T));
		CUDA_CHECK(cudaMemcpyAsync((void *)t, d_ptr, count*sizeof(T), cudaMemcpyDeviceToHost, stream));
	}

	size_t sizeInBytes { 0 };
	void  *d_ptr { nullptr };
};

}