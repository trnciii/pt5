#pragma once

#include <memory>
#include "data.h"

namespace pt5{
namespace material{

	enum class Type{
		Diffuse,
		Emission,
	};


	struct Material{
		virtual size_t size()const=0;
		virtual void* ptr()const=0;
		virtual Type type()const=0;
	};

	template <typename T>
	struct Material_t : Material{
		T data;

		Material_t():data(){}
		Material_t(const T& d):data(d){}

		size_t size() const{return sizeof(T);}
		void* ptr() const{return (void*)&data;}
		Type type()const;
	};


	template<>
	inline Type Material_t<BSDFData_Diffuse>::type()const{return Type::Diffuse;}

	template<>
	inline Type Material_t<BSDFData_Emission>::type()const{return Type::Emission;}


	template <typename T>
	inline std::shared_ptr<Material> abstract_material(const T& data){
		return std::make_shared<Material_t<T>>(Material_t(data));
	}


}

using MaterialType = material::Type;
using Material = material::Material;
using material::Material_t;
using material::abstract_material;


}
