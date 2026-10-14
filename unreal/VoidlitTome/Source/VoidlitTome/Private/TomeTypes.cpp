#include "TomeTypes.h"

#include "Misc/Parse.h"

namespace TomeJson
{
	FColor ParseHexColor(const FString& Hex, FColor Fallback)
	{
		FString H = Hex;
		H.RemoveFromStart(TEXT("#"));
		if (H.Len() != 6)
		{
			return Fallback;
		}
		uint8 Bytes[3];
		for (int32 i = 0; i < 3; ++i)
		{
			const TCHAR Hi = H[i * 2], Lo = H[i * 2 + 1];
			if (!FChar::IsHexDigit(Hi) || !FChar::IsHexDigit(Lo))
			{
				return Fallback;
			}
			Bytes[i] = static_cast<uint8>(FParse::HexDigit(Hi) * 16 + FParse::HexDigit(Lo));
		}
		return FColor(Bytes[0], Bytes[1], Bytes[2], 255);
	}

	ETomeEffect ParseEffect(const FString& Name)
	{
		if (Name == TEXT("none")) return ETomeEffect::None;
		if (Name == TEXT("pulse")) return ETomeEffect::Pulse;
		if (Name == TEXT("flicker")) return ETomeEffect::Flicker;
		if (Name == TEXT("ripple")) return ETomeEffect::Ripple;
		return ETomeEffect::Rise;
	}

	static void ParseSpans(const TArray<TSharedPtr<FJsonValue>>& In, TArray<FTomeSpan>& Out)
	{
		for (const TSharedPtr<FJsonValue>& V : In)
		{
			const TSharedPtr<FJsonObject>* SpanObj = nullptr;
			if (!V.IsValid() || !V->TryGetObject(SpanObj))
			{
				continue;
			}
			FTomeSpan Span;
			(*SpanObj)->TryGetStringField(TEXT("text"), Span.Text);
			FString Color, Effect;
			if ((*SpanObj)->TryGetStringField(TEXT("color"), Color))
			{
				Span.Color = ParseHexColor(Color);
			}
			double Glow = 0.0;
			if ((*SpanObj)->TryGetNumberField(TEXT("glow"), Glow))
			{
				Span.Glow = FMath::Clamp(static_cast<float>(Glow), 0.f, 1.f);
			}
			if ((*SpanObj)->TryGetStringField(TEXT("effect"), Effect))
			{
				Span.Effect = ParseEffect(Effect);
			}
			Out.Add(MoveTemp(Span));
		}
	}

	bool ParseFrame(const TSharedPtr<FJsonObject>& Obj, FTomeFrame& Out)
	{
		const TArray<TSharedPtr<FJsonValue>>* Blocks = nullptr;
		if (!Obj.IsValid() || !Obj->TryGetArrayField(TEXT("blocks"), Blocks))
		{
			return false;
		}
		Out = FTomeFrame();
		Obj->TryGetNumberField(TEXT("seq"), Out.Seq);
		Obj->TryGetBoolField(TEXT("over"), Out.bOver);

		for (const TSharedPtr<FJsonValue>& V : *Blocks)
		{
			const TSharedPtr<FJsonObject>* BlockObj = nullptr;
			if (!V.IsValid() || !V->TryGetObject(BlockObj))
			{
				continue;
			}
			FTomeBlock Block;
			(*BlockObj)->TryGetStringField(TEXT("role"), Block.Role);
			(*BlockObj)->TryGetStringField(TEXT("speaker"), Block.Speaker);
			const TArray<TSharedPtr<FJsonValue>>* Spans = nullptr;
			if ((*BlockObj)->TryGetArrayField(TEXT("spans"), Spans))
			{
				ParseSpans(*Spans, Block.Spans);
			}
			Out.Blocks.Add(MoveTemp(Block));
		}

		const TSharedPtr<FJsonObject>* Mood = nullptr;
		if (Obj->TryGetObjectField(TEXT("mood"), Mood))
		{
			double Value = 0.0;
			if ((*Mood)->TryGetNumberField(TEXT("lumen"), Value)) Out.Mood.Lumen = static_cast<float>(Value);
			if ((*Mood)->TryGetNumberField(TEXT("tension"), Value)) Out.Mood.Tension = static_cast<float>(Value);
			if ((*Mood)->TryGetNumberField(TEXT("stress"), Value)) Out.Mood.Stress = static_cast<float>(Value);
			(*Mood)->TryGetBoolField(TEXT("dark"), Out.Mood.bDark);
			FString Pulse;
			Out.Mood.bHeartbeat = (*Mood)->TryGetStringField(TEXT("pulse"), Pulse) && Pulse == TEXT("heartbeat");
		}

		const TSharedPtr<FJsonObject>* Status = nullptr;
		if (Obj->TryGetObjectField(TEXT("status"), Status))
		{
			const TArray<TSharedPtr<FJsonValue>>* Pair = nullptr;
			if ((*Status)->TryGetArrayField(TEXT("hull"), Pair) && Pair->Num() == 2)
			{
				Out.Status.Hull = static_cast<int32>((*Pair)[0]->AsNumber());
				Out.Status.HullMax = static_cast<int32>((*Pair)[1]->AsNumber());
			}
			if ((*Status)->TryGetArrayField(TEXT("lumen"), Pair) && Pair->Num() == 2)
			{
				Out.Status.Lumen = static_cast<int32>((*Pair)[0]->AsNumber());
				Out.Status.LumenMax = static_cast<int32>((*Pair)[1]->AsNumber());
			}
			(*Status)->TryGetNumberField(TEXT("stress"), Out.Status.Stress);
			(*Status)->TryGetBoolField(TEXT("won"), Out.Status.bWon);
			const TSharedPtr<FJsonObject>* Chapter = nullptr;
			if ((*Status)->TryGetObjectField(TEXT("chapter"), Chapter))
			{
				(*Chapter)->TryGetStringField(TEXT("title"), Out.Status.ChapterTitle);
				(*Chapter)->TryGetNumberField(TEXT("index"), Out.Status.ChapterIndex);
				(*Chapter)->TryGetNumberField(TEXT("total"), Out.Status.ChapterTotal);
			}
		}

		const TArray<TSharedPtr<FJsonValue>>* Effects = nullptr;
		if (Obj->TryGetArrayField(TEXT("effects"), Effects))
		{
			for (const TSharedPtr<FJsonValue>& V : *Effects)
			{
				const TSharedPtr<FJsonObject>* E = nullptr;
				if (V.IsValid() && V->TryGetObject(E))
				{
					FTomeEffectCue Cue;
					(*E)->TryGetStringField(TEXT("kind"), Cue.Kind);
					double Strength = 0.0;
					(*E)->TryGetNumberField(TEXT("strength"), Strength);
					Cue.Strength = static_cast<float>(Strength);
					Out.Effects.Add(Cue);
				}
			}
		}
		return true;
	}
}
