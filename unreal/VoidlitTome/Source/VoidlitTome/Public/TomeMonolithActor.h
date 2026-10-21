#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "HAL/PlatformProcess.h"
#include "TomeBrainClient.h"
#include "TomeTypesetter.h"

#include "TomeMonolithActor.generated.h"

class SEditableText;
class SWidget;
class UStaticMeshComponent;
class UTomeVoxelSurfaceComponent;

/**
 * The Voidlit Tome: a matte-black 1:4:9 slab whose face is a UTomeVoxelSurfaceComponent.
 *
 * Drop one in a level and press Play. It connects to the Tome's mind (`python -m voidlit serve`, optionally
 * spawned for you), captures the keyboard with an invisible text box, and renders everything (your typing,
 * the narration, NPC voices and status) as voxels rising out of the surface.
 *
 * Typed commands: ":save [slot]", ":load [slot]", ":new".
 */
UCLASS(Blueprintable)
class VOIDLITTOME_API ATomeMonolithActor : public AActor
{
	GENERATED_BODY()

public:
	ATomeMonolithActor();

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Tome")
	TObjectPtr<USceneComponent> Root;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Tome")
	TObjectPtr<UStaticMeshComponent> Slab;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Tome")
	TObjectPtr<UTomeVoxelSurfaceComponent> Surface;

	/** Slab thickness (cm). Width and height follow the voxel grid: Columns and Rows times the pitch. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Tome|Slab", meta = (ClampMin = 1.0))
	float SlabThicknessCm = 10.f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Tome|Brain")
	FString BrainUrl = TEXT("ws://127.0.0.1:8765");

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Tome|Brain")
	FTomeSessionOptions SessionOptions;

	/** Launch the Python sidecar on BeginPlay and stop it on EndPlay. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Tome|Brain")
	bool bSpawnBrain = false;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Tome|Brain", meta = (EditCondition = "bSpawnBrain"))
	FString PythonExecutable = TEXT("python");

	/** Add "--model <path>.gguf" to use a local llama.cpp model (loaded in the brain's process; no network). */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Tome|Brain", meta = (EditCondition = "bSpawnBrain"))
	FString BrainArguments = TEXT("-m voidlit serve --port 8765");

	/** Working directory for the sidecar (the repo's brain/ folder). */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Tome|Brain", meta = (EditCondition = "bSpawnBrain"))
	FString BrainWorkingDirectory;

	/** Capture the keyboard with an invisible text box so typing lands on the Tome. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Tome|Input")
	bool bCaptureKeyboard = true;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Tome|Input", meta = (ClampMin = 1))
	int32 MaxInputChars = 300;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	TObjectPtr<UTomeBrainClient> Brain;

	/** Submit a line as if typed (also handles ":save", ":load" and ":new"). */
	UFUNCTION(BlueprintCallable, Category = "Tome")
	void SubmitInput(const FString& Text);

	UFUNCTION(BlueprintCallable, Category = "Tome")
	void SetInputText(const FString& Text);

	virtual void OnConstruction(const FTransform& Transform) override;
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
	virtual void Tick(float DeltaSeconds) override;

private:
	UFUNCTION()
	void HandleFrame(const FTomeFrame& Frame);

	UFUNCTION()
	void HandleThinking();

	UFUNCTION()
	void HandleConnection(bool bConnected);

	void LayoutSlab();
	void Refresh(int32 NewTextFromSerial = -1);
	void CreateInputCatcher();
	void RemoveInputCatcher();
	void SpawnBrain();

	FTomeTypesetter Typesetter;
	FTomeStatus LastStatus;
	FString InputText;
	bool bConnected = false;

	TSharedPtr<SEditableText> InputWidget;
	TSharedPtr<SWidget> InputRoot;
	FProcHandle BrainProcess;
};
