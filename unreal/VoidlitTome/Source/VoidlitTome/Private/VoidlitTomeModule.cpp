#include "VoidlitTomeModule.h"

#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "ShaderCore.h"

DEFINE_LOG_CATEGORY(LogVoidlitTome);

void FVoidlitTomeModule::StartupModule()
{
	// Lets the material's Custom nodes include "/Plugin/VoidlitTome/TomeVoxel.ush".
	if (const TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("VoidlitTome")))
	{
		const FString ShaderDir = FPaths::Combine(Plugin->GetBaseDir(), TEXT("Shaders"));
		AddShaderSourceDirectoryMapping(TEXT("/Plugin/VoidlitTome"), ShaderDir);
	}
}

void FVoidlitTomeModule::ShutdownModule()
{
}

IMPLEMENT_MODULE(FVoidlitTomeModule, VoidlitTome)
