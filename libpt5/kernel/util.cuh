#pragma once

#include <optix_device.h>
#include <stdint.h>
#include <curand_kernel.h>


namespace pt5{

static __forceinline__ __device__
void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 ){
	const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__
void *unpackPointer(uint32_t i0, uint32_t i1){
	const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
	void* ptr = reinterpret_cast<void*>(uptr);
	return ptr;
}

template<typename T>
static __forceinline__ __device__
T* getPRD(){
	const uint32_t u0 = optixGetPayload_0();
	const uint32_t u1 = optixGetPayload_1();
	return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

} // pt5