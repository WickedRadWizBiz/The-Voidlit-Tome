using UnrealBuildTool;

public class VoidlitTome : ModuleRules
{
	public VoidlitTome(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core", "CoreUObject", "Engine", "InputCore"
		});

		PrivateDependencyModuleNames.AddRange(new string[]
		{
			"Projects", "RenderCore", "RHI", "Json", "WebSockets", "Slate", "SlateCore"
		});
	}
}
