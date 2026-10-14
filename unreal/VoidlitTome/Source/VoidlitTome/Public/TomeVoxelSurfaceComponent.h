#pragma once

#include "Components/InstancedStaticMeshComponent.h"
#include "CoreMinimal.h"
#include "TomeTypesetter.h"

#include "TomeVoxelSurfaceComponent.generated.h"

class UMaterialInstanceDynamic;
class UMaterialInterface;
class UTexture2D;

/**
 * The Tome's face: a dense grid of machined voxels. Flush, they are invisible in the matte black slab;
 * raised by RiseCm, they glow in colour and become text.
 *
 * Every voxel is one instance of a tiny cube. Its column and row live in PerInstanceCustomData[0..1], and two
 * textures hold one texel per voxel (see Shaders/TomeVoxel.ush):
 *   TomeState  RGBA32F  start time, height before, glow, height after
 *   TomeColor  BGRA8    sRGB colour, effect id
 * The GPU animates every transition, so the CPU only rewrites the textures when the page changes.
 *
 * Local frame: the face lies in the component's YZ plane and faces +X; voxel tops sit at X = 0.
 */
UCLASS(ClassGroup = (VoidlitTome), meta = (BlueprintSpawnableComponent))
class VOIDLITTOME_API UTomeVoxelSurfaceComponent : public UInstancedStaticMeshComponent
{
	GENERATED_BODY()

public:
	UTomeVoxelSurfaceComponent();

	/** Voxels across the face. 320 x 720 at 1.25 mm fills a 40 x 90 cm monolith face (the 1:4:9 slab). */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Tome|Grid", meta = (ClampMin = 32, ClampMax = 1024))
	int32 Columns = 320;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Tome|Grid", meta = (ClampMin = 32, ClampMax = 2048))
	int32 Rows = 720;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Tome|Grid", meta = (ClampMin = 0.01))
	float VoxelPitchCm = 0.125f;

	/** How far a fully raised voxel stands proud of the surface (about 1 mm). */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Tome|Grid", meta = (ClampMin = 0.0))
	float RiseCm = 0.1f;

	/** Voxel body depth; must exceed RiseCm so a raised voxel never shows a gap beneath it. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Tome|Grid", meta = (ClampMin = 0.01))
	float VoxelDepthCm = 0.3f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Tome|Layout", meta = (ClampMin = 0))
	int32 MarginVoxels = 8;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Tome|Layout", meta = (ClampMin = 0))
	int32 LineGapVoxels = 2;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Tome|Animation", meta = (ClampMin = 0.01))
	float RiseSeconds = 0.35f;

	/** Typewriter stagger between consecutive new characters. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Tome|Animation", meta = (ClampMin = 0.0))
	float CharDelaySeconds = 0.006f;

	/** Cap on the typewriter stagger so long passages finish rising quickly. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Tome|Animation", meta = (ClampMin = 0.0))
	float MaxTypewriterSeconds = 3.5f;

	/** M_TomeVoxel (created by Content/Python/create_tome_material.py). Falls back to /VoidlitTome/M_TomeVoxel. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Tome|Look")
	TObjectPtr<UMaterialInterface> VoxelMaterial;

	UFUNCTION(BlueprintCallable, Category = "Tome")
	void BuildGrid();

	/** Character cells that fit on the face with the current margins and glyph size. */
	UFUNCTION(BlueprintPure, Category = "Tome")
	int32 GetTextColumns() const;

	UFUNCTION(BlueprintPure, Category = "Tome")
	int32 GetTextRows() const;

	/** Retargets every voxel for a new page. Cells with Serial >= NewTextFromSerial rise in typewriter order. */
	void ShowPage(const TArray<FTomeLine>& Page, int32 NewTextFromSerial);

	UFUNCTION(BlueprintCallable, Category = "Tome")
	void SetMood(float InLumen, bool bInDark, bool bInHeartbeat);

	UFUNCTION(BlueprintCallable, Category = "Tome")
	void SetThinking(bool bInThinking);

	/** shudder | ripple | flare | panic | dim */
	UFUNCTION(BlueprintCallable, Category = "Tome")
	void PlayCue(const FString& Kind, float Strength);

	virtual void BeginPlay() override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

private:
	float TomeTime() const;
	float EvalHeight(const FLinearColor& S, float Time) const;
	void CreateTextures();
	void UploadTextures();
	void PushParameters();

	UPROPERTY(Transient)
	TObjectPtr<UTexture2D> StateTexture;

	UPROPERTY(Transient)
	TObjectPtr<UTexture2D> ColorTexture;

	UPROPERTY(Transient)
	TObjectPtr<UMaterialInstanceDynamic> SurfaceMID;

	TArray<FLinearColor> State;
	TArray<FColor> Colors;
	bool bBuilt = false;

	float Lumen = 1.f, TargetLumen = 1.f;
	float Dark = 0.f, TargetDark = 0.f;
	float Thinking = 0.f, TargetThinking = 0.f;
	float Heartbeat = 0.f, TargetHeartbeat = 0.f;
	float RippleTime = -100.f, RippleStrength = 0.f;
	float Shudder = 0.f, Panic = 0.f, Flare = 0.f;
};
