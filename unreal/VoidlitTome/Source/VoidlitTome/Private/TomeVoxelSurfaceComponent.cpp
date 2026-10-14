#include "TomeVoxelSurfaceComponent.h"

#include "Engine/StaticMesh.h"
#include "Engine/Texture2D.h"
#include "Engine/World.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Materials/MaterialInterface.h"
#include "RenderUtils.h"
#include "UObject/ConstructorHelpers.h"
#include "VoidlitTomeModule.h"

namespace TomeParam
{
	static const FName StateTex(TEXT("TomeState"));
	static const FName ColorTex(TEXT("TomeColor"));
	static const FName Time(TEXT("TomeTime"));
	static const FName GridCols(TEXT("GridCols"));
	static const FName GridRows(TEXT("GridRows"));
	static const FName RiseCm(TEXT("RiseCm"));
	static const FName FaceNormal(TEXT("FaceNormalWS"));
	static const FName RiseSeconds(TEXT("RiseSeconds"));
	static const FName RippleTime(TEXT("RippleTime"));
	static const FName RippleStrength(TEXT("RippleStrength"));
	static const FName Panic(TEXT("Panic"));
	static const FName Thinking(TEXT("Thinking"));
	static const FName Shudder(TEXT("Shudder"));
	static const FName Lumen(TEXT("Lumen"));
	static const FName Dark(TEXT("Dark"));
	static const FName Heartbeat(TEXT("Heartbeat"));
	static const FName Flare(TEXT("Flare"));
}

UTomeVoxelSurfaceComponent::UTomeVoxelSurfaceComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	PrimaryComponentTick.TickGroup = TG_PostUpdateWork;

	static ConstructorHelpers::FObjectFinder<UStaticMesh> Cube(TEXT("/Engine/BasicShapes/Cube.Cube"));
	if (Cube.Succeeded())
	{
		SetStaticMesh(Cube.Object);
	}
	NumCustomDataFloats = 2;
	SetCollisionEnabled(ECollisionEnabled::NoCollision);
	SetCanEverAffectNavigation(false);
	// Hundreds of thousands of millimetre voxels: shadows are opt-in.
	CastShadow = false;
}

int32 UTomeVoxelSurfaceComponent::GetTextColumns() const
{
	return FMath::Max(1, (Columns - 2 * MarginVoxels) / FTomeGlyphFont::Width);
}

int32 UTomeVoxelSurfaceComponent::GetTextRows() const
{
	return FMath::Max(1, (Rows - 2 * MarginVoxels) / (FTomeGlyphFont::Height + LineGapVoxels));
}

float UTomeVoxelSurfaceComponent::TomeTime() const
{
	const UWorld* World = GetWorld();
	return World ? World->GetTimeSeconds() : 0.f;
}

float UTomeVoxelSurfaceComponent::EvalHeight(const FLinearColor& S, float Time) const
{
	// Mirrors TomeEvalVoxel in TomeVoxel.ush.
	const float T = FMath::Clamp((Time - S.R) / FMath::Max(RiseSeconds, 0.001f), 0.f, 1.f);
	return FMath::Lerp(S.G, S.A, T * T * (3.f - 2.f * T));
}

void UTomeVoxelSurfaceComponent::BeginPlay()
{
	Super::BeginPlay();
	if (!bBuilt)
	{
		BuildGrid();
	}
}

void UTomeVoxelSurfaceComponent::BuildGrid()
{
	ClearInstances();
	SetNumCustomDataFloats(2);

	const int32 N = Columns * Rows;
	const float MeshSize = 100.f;  // /Engine/BasicShapes/Cube is 100 cm
	const FVector Scale(VoxelDepthCm / MeshSize, VoxelPitchCm / MeshSize, VoxelPitchCm / MeshSize);

	TArray<FTransform> Transforms;
	Transforms.Reserve(N);
	for (int32 Row = 0; Row < Rows; ++Row)
	{
		for (int32 Col = 0; Col < Columns; ++Col)
		{
			const FVector Location(-0.5f * VoxelDepthCm, (Col - 0.5f * Columns + 0.5f) * VoxelPitchCm,
				(0.5f * Rows - Row - 0.5f) * VoxelPitchCm);
			Transforms.Emplace(FQuat::Identity, Location, Scale);
		}
	}
	AddInstances(Transforms, /*bShouldReturnIndices*/ false, /*bWorldSpace*/ false);
	for (int32 i = 0; i < N; ++i)
	{
		SetCustomDataValue(i, 0, static_cast<float>(i % Columns), false);
		SetCustomDataValue(i, 1, static_cast<float>(i / Columns), false);
	}
	MarkRenderStateDirty();

	State.Init(FLinearColor(-100.f, 0.f, 0.f, 0.f), N);
	Colors.Init(FColor(0, 0, 0, 0), N);
	CreateTextures();

	UMaterialInterface* Base = VoxelMaterial;
	if (!Base)
	{
		Base = LoadObject<UMaterialInterface>(nullptr, TEXT("/VoidlitTome/M_TomeVoxel.M_TomeVoxel"));
	}
	if (Base)
	{
		SurfaceMID = UMaterialInstanceDynamic::Create(Base, this);
		SurfaceMID->SetTextureParameterValue(TomeParam::StateTex, StateTexture);
		SurfaceMID->SetTextureParameterValue(TomeParam::ColorTex, ColorTexture);
		SetMaterial(0, SurfaceMID);
	}
	else
	{
		UE_LOG(LogVoidlitTome, Warning,
			TEXT("No voxel material: run Content/Python/create_tome_material.py in the editor, or assign VoxelMaterial."));
	}
	bBuilt = true;
	UploadTextures();
	PushParameters();
}

void UTomeVoxelSurfaceComponent::CreateTextures()
{
	auto Make = [this](EPixelFormat Format, const TCHAR* Name)
	{
		UTexture2D* Tex = UTexture2D::CreateTransient(Columns, Rows, Format, FName(Name));
		Tex->Filter = TF_Nearest;
		Tex->SRGB = false;
		Tex->NeverStream = true;
		Tex->AddressX = TA_Clamp;
		Tex->AddressY = TA_Clamp;
		Tex->UpdateResource();
		return Tex;
	};
	StateTexture = Make(PF_A32B32G32R32F, TEXT("TomeState"));
	ColorTexture = Make(PF_B8G8R8A8, TEXT("TomeColor"));
}

void UTomeVoxelSurfaceComponent::UploadTextures()
{
	auto Upload = [this](UTexture2D* Tex, const void* Source, int32 BytesPerPixel)
	{
		if (!Tex)
		{
			return;
		}
		const int32 Size = Columns * Rows * BytesPerPixel;
		uint8* Copy = static_cast<uint8*>(FMemory::Malloc(Size));
		FMemory::Memcpy(Copy, Source, Size);
		FUpdateTextureRegion2D* Region = new FUpdateTextureRegion2D(0, 0, 0, 0, Columns, Rows);
		Tex->UpdateTextureRegions(0, 1, Region, Columns * BytesPerPixel, BytesPerPixel, Copy,
			[](uint8* Data, const FUpdateTextureRegion2D* Regions)
			{
				FMemory::Free(Data);
				delete Regions;
			});
	};
	Upload(StateTexture, State.GetData(), sizeof(FLinearColor));
	Upload(ColorTexture, Colors.GetData(), sizeof(FColor));
}

void UTomeVoxelSurfaceComponent::ShowPage(const TArray<FTomeLine>& Page, int32 NewTextFromSerial)
{
	if (!bBuilt)
	{
		BuildGrid();
	}
	const int32 N = Columns * Rows;
	const float Now = TomeTime();

	TArray<float> To;
	TArray<float> Glow;
	TArray<float> Delay;
	TArray<FColor> Want;
	To.Init(0.f, N);
	Glow.Init(0.f, N);
	Delay.Init(0.f, N);
	Want.Init(FColor(0, 0, 0, 0), N);

	// Rasterize the page's characters into voxel targets.
	const int32 TextCols = GetTextColumns();
	const int32 TextRows = GetTextRows();
	const int32 LineHeight = FTomeGlyphFont::Height + LineGapVoxels;
	for (int32 L = 0; L < FMath::Min(TextRows, Page.Num()); ++L)
	{
		const FTomeLine& Line = Page[L];
		for (int32 C = 0; C < FMath::Min(TextCols, Line.Num()); ++C)
		{
			const FTomeCell& Cell = Line[C];
			if (Cell.Char == TEXT(' ') || Cell.Height <= 0.f)
			{
				continue;
			}
			const uint8* Glyph = FTomeGlyphFont::GetRows(static_cast<uint32>(Cell.Char));
			const int32 X0 = MarginVoxels + C * FTomeGlyphFont::Width;
			const int32 Y0 = MarginVoxels + L * LineHeight;
			const float CellDelay = Cell.Serial >= NewTextFromSerial && NewTextFromSerial >= 0
				? FMath::Min((Cell.Serial - NewTextFromSerial) * CharDelaySeconds, MaxTypewriterSeconds)
				: 0.f;
			FColor Packed = Cell.Color;
			Packed.A = static_cast<uint8>(Cell.Effect);
			for (int32 GY = 0; GY < FTomeGlyphFont::Height; ++GY)
			{
				const uint8 Bits = Glyph[GY];
				if (!Bits)
				{
					continue;
				}
				for (int32 GX = 0; GX < FTomeGlyphFont::Width; ++GX)
				{
					if (Bits & (0x80 >> GX))
					{
						const int32 X = X0 + GX, Y = Y0 + GY;
						if (X < Columns && Y < Rows)
						{
							const int32 i = Y * Columns + X;
							To[i] = Cell.Height;
							Glow[i] = Cell.Glow;
							Delay[i] = CellDelay;
							Want[i] = Packed;
						}
					}
				}
			}
		}
	}

	// Reconcile against what is on the surface now: only changed voxels start a new transition.
	for (int32 i = 0; i < N; ++i)
	{
		FLinearColor& S = State[i];
		if (To[i] > 0.f)
		{
			const bool bSameHeight = FMath::IsNearlyEqual(To[i], S.A, 1e-3f);
			const bool bSameLook = Colors[i] == Want[i] && FMath::IsNearlyEqual(Glow[i], S.B, 1e-3f);
			if (bSameHeight && bSameLook)
			{
				continue;
			}
			if (bSameHeight && S.A > 0.f)
			{
				// Same shape, new colour: recolour in place without re-rising.
				Colors[i] = Want[i];
				S.B = Glow[i];
				continue;
			}
			S = FLinearColor(Now + Delay[i], EvalHeight(S, Now), Glow[i], To[i]);
			Colors[i] = Want[i];
		}
		else if (S.A > 0.f)
		{
			// Sink, keeping colour and glow while it descends. A little jitter makes it feel machined, not printed.
			S = FLinearColor(Now + FMath::FRand() * 0.12f, EvalHeight(S, Now), S.B, 0.f);
		}
	}
	UploadTextures();
}

void UTomeVoxelSurfaceComponent::SetMood(float InLumen, bool bInDark, bool bInHeartbeat)
{
	TargetLumen = FMath::Clamp(InLumen, 0.f, 1.f);
	TargetDark = bInDark ? 1.f : 0.f;
	TargetHeartbeat = bInHeartbeat ? 1.f : 0.f;
}

void UTomeVoxelSurfaceComponent::SetThinking(bool bInThinking)
{
	TargetThinking = bInThinking ? 1.f : 0.f;
}

void UTomeVoxelSurfaceComponent::PlayCue(const FString& Kind, float Strength)
{
	const float S = FMath::Clamp(Strength, 0.f, 1.f);
	if (Kind == TEXT("shudder"))
	{
		Shudder = FMath::Max(Shudder, S);
	}
	else if (Kind == TEXT("ripple"))
	{
		RippleTime = TomeTime();
		RippleStrength = FMath::Max(0.3f, S);
	}
	else if (Kind == TEXT("flare"))
	{
		Flare = FMath::Max(Flare, S);
	}
	else if (Kind == TEXT("panic"))
	{
		Panic = 1.f;
	}
	else if (Kind == TEXT("dim"))
	{
		Shudder = FMath::Max(Shudder, 0.2f * S);
	}
}

void UTomeVoxelSurfaceComponent::PushParameters()
{
	if (!SurfaceMID)
	{
		return;
	}
	SurfaceMID->SetScalarParameterValue(TomeParam::Time, TomeTime());
	SurfaceMID->SetScalarParameterValue(TomeParam::GridCols, static_cast<float>(Columns));
	SurfaceMID->SetScalarParameterValue(TomeParam::GridRows, static_cast<float>(Rows));
	SurfaceMID->SetScalarParameterValue(TomeParam::RiseCm, RiseCm * GetComponentScale().X);
	const FVector Normal = GetForwardVector();
	SurfaceMID->SetVectorParameterValue(TomeParam::FaceNormal, FLinearColor(Normal.X, Normal.Y, Normal.Z, 0.f));
	SurfaceMID->SetScalarParameterValue(TomeParam::RiseSeconds, RiseSeconds);
	SurfaceMID->SetScalarParameterValue(TomeParam::RippleTime, RippleTime);
	SurfaceMID->SetScalarParameterValue(TomeParam::RippleStrength, RippleStrength);
	SurfaceMID->SetScalarParameterValue(TomeParam::Panic, Panic);
	SurfaceMID->SetScalarParameterValue(TomeParam::Thinking, Thinking);
	SurfaceMID->SetScalarParameterValue(TomeParam::Shudder, Shudder);
	SurfaceMID->SetScalarParameterValue(TomeParam::Lumen, Lumen);
	SurfaceMID->SetScalarParameterValue(TomeParam::Dark, Dark);
	SurfaceMID->SetScalarParameterValue(TomeParam::Heartbeat, Heartbeat);
	SurfaceMID->SetScalarParameterValue(TomeParam::Flare, Flare);
}

void UTomeVoxelSurfaceComponent::TickComponent(float DeltaTime, ELevelTick TickType,
	FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	Lumen = FMath::FInterpTo(Lumen, TargetLumen, DeltaTime, 1.5f);
	Dark = FMath::FInterpTo(Dark, TargetDark, DeltaTime, 0.8f);
	Thinking = FMath::FInterpTo(Thinking, TargetThinking, DeltaTime, 3.f);
	Heartbeat = FMath::FInterpTo(Heartbeat, TargetHeartbeat, DeltaTime, 2.f);
	Shudder *= FMath::Exp(-4.f * DeltaTime);
	Panic *= FMath::Exp(-2.5f * DeltaTime);
	Flare *= FMath::Exp(-3.f * DeltaTime);
	PushParameters();
}
