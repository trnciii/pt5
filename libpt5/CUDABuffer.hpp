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

	void alloc_and_upload(const void* data, size_t size, CUstream stream){
		assert(d_ptr == nullptr);
		alloc(size, stream);
		CUDA_CHECK(cudaMemcpyAsync(d_ptr, data, size, cudaMemcpyHostToDevice, stream));
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

	size_t size()const{return sizeInBytes;}

	bool allocated()const{return d_ptr != nullptr;}

private:
	size_t sizeInBytes { 0 };
	void  *d_ptr { nullptr };
};


template<typename T>
struct Array2{
	uint2 size = {0,0};
	cudaArray_t array = nullptr;

	~Array2(){
		if(array != 0) cudaFreeArray(array);
	}

	void alloc(uint w, uint h){
		assert(array == nullptr);
		cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<T>();
		CUDA_CHECK(cudaMallocArray(&array, &channelFormatDesc, w, h));
		size = {w, h};
	}

	void upload(const std::vector<T>& pixels, CUstream stream){
		assert(pixels.size() == (size.x*size.y));
		upload(pixels.data(), stream);
	}

	void upload(const T* data, CUstream stream){
		assert(array != nullptr);

		uint32_t pitch = size.x * sizeof(T);
		CUDA_CHECK(cudaMemcpy2DToArrayAsync(
			array,
			0, 0,
			data,
			pitch,
			pitch, size.y,
			cudaMemcpyDefault,
			stream
		));
	}

	void alloc_and_upload(uint w, uint h, const std::vector<T>& pixels, CUstream stream){
		alloc(w, h);
		upload(pixels, stream);
	}

	void alloc_and_upload(uint w, uint h, const T* data, CUstream stream){
		alloc(w, h);
		upload(data, stream);
	}

	void free(){
		assert(array != nullptr);
		CUDA_CHECK(cudaFreeArray(array));
		size = {0,0};
	}

	void download(T *t, size_t count, CUstream stream){
		assert(array != nullptr);
		assert((size.x*size.y) == count);

		uint32_t pitch = size.x * sizeof(T);
		CUDA_CHECK(cudaMemcpy2DFromArrayAsync(
			t,
			pitch,
			array,
			0, 0,
			pitch, size.y,
			cudaMemcpyDefault,
			stream
		));
	}
};

using Image = Array2<float4>;

}