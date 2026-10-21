#pragma once

#include "CoreMinimal.h"
#include "Dom/JsonObject.h"

#include "TomeTypes.generated.h"

/** Effect hints carried by spans (see docs/PROTOCOL.md). Values match TOME_FX_* in TomeVoxel.ush. */
UENUM(BlueprintType)
enum class ETomeEffect : uint8
{
	None = 0,
	Rise = 1,
	Pulse = 2,
	Flicker = 3,
	Ripple = 4,
};

USTRUCT(BlueprintType)
struct VOIDLITTOME_API FTomeSpan
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	FString Text;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	FColor Color = FColor(201, 209, 224);

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	float Glow = 0.f;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	ETomeEffect Effect = ETomeEffect::Rise;
};

USTRUCT(BlueprintType)
struct VOIDLITTOME_API FTomeBlock
{
	GENERATED_BODY()

	/** title | echo | narration | npc | prompt | system */
	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	FString Role;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	FString Speaker;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	TArray<FTomeSpan> Spans;
};

USTRUCT(BlueprintType)
struct VOIDLITTOME_API FTomeMood
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	float Lumen = 1.f;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	float Stress = 0.f;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	bool bDark = false;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	bool bHeartbeat = false;
};

USTRUCT(BlueprintType)
struct VOIDLITTOME_API FTomeEffectCue
{
	GENERATED_BODY()

	/** shudder | dim | flare | panic | ripple */
	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	FString Kind;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	float Strength = 0.f;
};

USTRUCT(BlueprintType)
struct VOIDLITTOME_API FTomeStatus
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	int32 Hull = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	int32 HullMax = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	int32 Stress = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	int32 Lumen = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	int32 LumenMax = 6;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	FString ChapterTitle;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	int32 ChapterIndex = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	int32 ChapterTotal = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	bool bWon = false;
};

USTRUCT(BlueprintType)
struct VOIDLITTOME_API FTomeFrame
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	int32 Seq = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	TArray<FTomeBlock> Blocks;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	FTomeMood Mood;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	FTomeStatus Status;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	TArray<FTomeEffectCue> Effects;

	UPROPERTY(BlueprintReadOnly, Category = "Tome")
	bool bOver = false;
};

namespace TomeJson
{
	/** "#rrggbb" -> FColor (sRGB bytes). */
	VOIDLITTOME_API FColor ParseHexColor(const FString& Hex, FColor Fallback = FColor(201, 209, 224));

	VOIDLITTOME_API ETomeEffect ParseEffect(const FString& Name);

	/** Parses a {"type": "frame", ...} message. Returns false if required fields are missing. */
	VOIDLITTOME_API bool ParseFrame(const TSharedPtr<FJsonObject>& Obj, FTomeFrame& Out);
}
