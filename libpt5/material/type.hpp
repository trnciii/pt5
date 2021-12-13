#pragma once

#include "data.h"

namespace pt5{

enum class MaterialType{
	Diffuse,
	Emission,
};


inline MaterialType dataType(const MTLData_Diffuse& t){
	return MaterialType::Diffuse;
}

inline MaterialType dataType(const MTLData_Emission& t){
	return MaterialType::Emission;
}


struct Material{
	virtual size_t size()const=0;
	virtual void* ptr()const=0;
	virtual MaterialType type()const=0;
};

template <typename T>
struct Material_t : Material{
	T data;

	Material_t():data(){}
	Material_t(const T& d):data(d){}

	size_t size() const{return sizeof(T);}
	void* ptr() const{return (void*)&data;}
	MaterialType type()const{return dataType(data);}
};

}
