#pragma once

#include <memory>
#include "data.h"

namespace pt5{

enum class MaterialType{
	Diffuse,
	Emission,
};


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
	MaterialType type()const;
};


template<>
inline MaterialType Material_t<MTLData_Diffuse>::type()const{return MaterialType::Diffuse;}

template<>
inline MaterialType Material_t<MTLData_Emission>::type()const{return MaterialType::Emission;}


template <typename T>
inline std::shared_ptr<Material> abstract_material(const T& data){
	return std::make_shared<Material_t<T>>(Material_t(data));
}


}
