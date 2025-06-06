#pragma once

#include "camera.hpp"
#include "material/data.h"
#include "material/node.hpp"
#include "material/type.hpp"
#include "view.hpp"
#include "tracer.hpp"
#include "util.hpp"

namespace pt5{
	using material::make_node;

	using Diffuse = material::DiffuseData;
	using Glossy = material::GlossyData;
	using Emission = material::EmissionData;
	using Mix = material::MixData;
	using Background = material::BackgroundData;
	using Texture = material::TextureCreateInfo;
	using TexType = material::TextureCreateInfo::Type;
};
