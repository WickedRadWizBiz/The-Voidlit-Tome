#include "TomeTypesetter.h"

#include "TomeGlyphData.inl"

static_assert(GTomeGlyphWidth == FTomeGlyphFont::Width && GTomeGlyphHeight == FTomeGlyphFont::Height,
	"Regenerate TomeGlyphData.inl for the current glyph size");

namespace
{
	const FColor StatusColor(111, 122, 143);
	const FColor LumenColor(255, 233, 168);
	const FColor HullColor(159, 230, 160);
	const FColor StressColor(255, 77, 109);
	const FColor InputColor(232, 232, 232);
}

// ------------------------------------------------------------------------------------------------ glyphs

const uint8* FTomeGlyphFont::GetRows(uint32 Codepoint)
{
	static const TMap<uint32, const uint8*> Table = []
	{
		TMap<uint32, const uint8*> T;
		for (const FTomeGlyphRow& G : GTomeGlyphs)
		{
			T.Add(G.Codepoint, G.Rows);
		}
		return T;
	}();
	if (const uint8* const* Found = Table.Find(Codepoint))
	{
		return *Found;
	}
	return Table.FindChecked(static_cast<uint32>('?'));
}

// ------------------------------------------------------------------------------------------------ typesetter

FTomeTypesetter::FTomeTypesetter(int32 InColumns, int32 InMaxLines)
	: Columns(FMath::Max(8, InColumns))
	, MaxLines(FMath::Max(32, InMaxLines))
{
}

void FTomeTypesetter::SetColumns(int32 InColumns)
{
	Columns = FMath::Max(8, InColumns);
	Clear();
}

void FTomeTypesetter::Clear()
{
	Lines.Reset();
	PromptSpans.Reset();
}

int32 FTomeTypesetter::AppendFrame(const FTomeFrame& Frame)
{
	const int32 First = NextSerial;
	PromptSpans.Reset();
	for (const FTomeBlock& Block : Frame.Blocks)
	{
		AppendBlock(Block);
	}
	Trim();
	return First;
}

void FTomeTypesetter::AppendBlock(const FTomeBlock& Block)
{
	if (Block.Role == TEXT("prompt"))
	{
		PromptSpans = Block.Spans;  // shown above the input line, not kept in the log
	}
	else if (Block.Role == TEXT("title"))
	{
		if (Lines.Num() > 0)
		{
			AppendBlank();
		}
		AppendWrapped(Block.Spans, 0, 0, true, 1.f);
		AppendBlank();
	}
	else if (Block.Role == TEXT("echo"))
	{
		AppendBlank();
		AppendWrapped(Block.Spans, 0, 2, false, 0.55f);
	}
	else if (Block.Role == TEXT("narration"))
	{
		AppendWrapped(Block.Spans, 0, 0, false, 1.f);
		AppendBlank();
	}
	else if (Block.Role == TEXT("npc"))
	{
		AppendWrapped(Block.Spans, 0, 2, false, 1.f);
		AppendBlank();
	}
	else  // system and anything unknown
	{
		AppendWrapped(Block.Spans, 1, 3, false, 0.8f);
	}
}

FTomeCell FTomeTypesetter::MakeCell(TCHAR C, const FTomeSpan& Span, float Height)
{
	FTomeCell Cell;
	Cell.Char = C;
	Cell.Color = Span.Color;
	Cell.Glow = Span.Glow;
	Cell.Effect = Span.Effect;
	Cell.Height = Height;
	if (C != TEXT(' '))
	{
		Cell.Serial = NextSerial++;
	}
	return Cell;
}

void FTomeTypesetter::AppendWrapped(const TArray<FTomeSpan>& Spans, int32 FirstIndent, int32 RestIndent, bool bCenter,
	float Height)
{
	// Flatten spans into styled cells split into paragraphs and words.
	TArray<TArray<FTomeLine>> Paragraphs;  // paragraph -> words -> cells
	Paragraphs.AddDefaulted();
	FTomeLine Word;
	auto EndWord = [&]()
	{
		if (Word.Num() > 0)
		{
			Paragraphs.Last().Add(MoveTemp(Word));
			Word.Reset();
		}
	};
	for (const FTomeSpan& Span : Spans)
	{
		for (const TCHAR C : Span.Text)
		{
			if (C == TEXT('\n'))
			{
				EndWord();
				Paragraphs.AddDefaulted();
			}
			else if (C == TEXT(' ') || C == TEXT('\t') || C == TEXT('\r'))
			{
				EndWord();
			}
			else
			{
				Word.Add(MakeCell(C, Span, Height));
			}
		}
	}
	EndWord();

	bool bFirstLine = true;
	FTomeLine Current;
	auto Flush = [&]()
	{
		const int32 Indent = bFirstLine ? FirstIndent : RestIndent;
		FTomeLine Line;
		const int32 Pad = bCenter ? FMath::Max(0, (Columns - Current.Num()) / 2) : Indent;
		Line.AddDefaulted(Pad);
		for (FTomeCell& Cell : Line)
		{
			Cell.Height = 0.f;
		}
		Line.Append(Current);
		Lines.Add(MoveTemp(Line));
		Current.Reset();
		bFirstLine = false;
	};

	for (int32 P = 0; P < Paragraphs.Num(); ++P)
	{
		if (P > 0)
		{
			if (Current.Num() > 0)
			{
				Flush();
			}
			AppendBlank();
		}
		for (FTomeLine& W : Paragraphs[P])
		{
			const int32 Width = FMath::Max(4, Columns - (bFirstLine ? FirstIndent : RestIndent));
			// Hard-split words longer than a line.
			while (W.Num() > Width)
			{
				if (Current.Num() > 0)
				{
					Flush();
				}
				Current.Append(W.GetData(), Width);
				W.RemoveAt(0, Width);
				Flush();
			}
			if (Current.Num() == 0)
			{
				Current.Append(W);
			}
			else if (Current.Num() + 1 + W.Num() <= Width)
			{
				FTomeCell Space = W[0];
				Space.Char = TEXT(' ');
				Space.Serial = -1;
				Current.Add(Space);
				Current.Append(W);
			}
			else
			{
				Flush();
				Current.Append(W);
			}
		}
		if (Current.Num() > 0)
		{
			Flush();
		}
	}
}

void FTomeTypesetter::AppendBlank()
{
	if (Lines.Num() > 0 && Lines.Last().Num() == 0)
	{
		return;  // never stack blank lines
	}
	Lines.AddDefaulted();
}

void FTomeTypesetter::Trim()
{
	if (Lines.Num() > MaxLines)
	{
		Lines.RemoveAt(0, Lines.Num() - MaxLines);
	}
}

void FTomeTypesetter::WriteText(FTomeLine& Line, int32 Col, const FString& Text, FColor Color, float Glow,
	ETomeEffect Effect, float Height) const
{
	for (int32 i = 0; i < Text.Len() && Col + i < Columns; ++i)
	{
		if (Col + i < 0)
		{
			continue;
		}
		if (Line.Num() <= Col + i)
		{
			const int32 Old = Line.Num();
			Line.AddDefaulted(Col + i + 1 - Old);
			for (int32 k = Old; k < Line.Num(); ++k)
			{
				Line[k].Height = 0.f;
			}
		}
		FTomeCell& Cell = Line[Col + i];
		Cell.Char = Text[i];
		Cell.Color = Color;
		Cell.Glow = Glow;
		Cell.Effect = Effect;
		Cell.Height = Height;
		Cell.Serial = -1;
	}
}

void FTomeTypesetter::ComposePage(int32 Rows, const FTomeStatus& Status, const FString& Input, bool bCursorOn,
	TArray<FTomeLine>& OutPage) const
{
	OutPage.Reset();
	OutPage.SetNum(FMath::Max(Rows, 6));
	const int32 R = OutPage.Num();

	// Status bar.
	FTomeLine& Bar = OutPage[0];
	int32 Col = 0;
	const FString Hull = FString::Printf(TEXT("HULL %d/%d"), Status.Hull, Status.HullMax);
	WriteText(Bar, Col, Hull, HullColor, 0.3f);
	Col += Hull.Len() + 2;
	const FString Stress = FString::Printf(TEXT("STRESS %d"), Status.Stress);
	WriteText(Bar, Col, Stress, StressColor, Status.Stress >= 4 ? 0.6f : 0.2f,
		Status.Stress >= 7 ? ETomeEffect::Pulse : ETomeEffect::None);
	Col += Stress.Len() + 2;
	FString Lumen;
	for (int32 i = 0; i < Status.LumenMax; ++i)
	{
		Lumen.AppendChar(i < Status.Lumen ? TCHAR(0x25CF) : TCHAR(0x00B7));
	}
	WriteText(Bar, Col, Lumen, LumenColor, 0.7f);
	if (Status.ChapterTotal > 0)
	{
		const FString Where = FString::Printf(TEXT("%d/%d"), Status.ChapterIndex + 1, Status.ChapterTotal);
		WriteText(Bar, Columns - Where.Len(), Where, StatusColor, 0.1f);
	}

	// Rule.
	FString Rule;
	for (int32 i = 0; i < Columns; i += 2)
	{
		Rule += TEXT("\u00b7 ");
	}
	WriteText(OutPage[1], 0, Rule, StatusColor, 0.f, ETomeEffect::None, 0.3f);

	// Prompt and input.
	FTomeLine& Prompt = OutPage[R - 2];
	Col = 0;
	for (const FTomeSpan& Span : PromptSpans)
	{
		WriteText(Prompt, Col, Span.Text, Span.Color, Span.Glow, Span.Effect, 0.8f);
		Col += Span.Text.Len();
	}
	FString Shown = TEXT("> ") + Input.Right(FMath::Max(0, Columns - 3));
	WriteText(OutPage[R - 1], 0, Shown, InputColor, 0.25f, ETomeEffect::None, 0.85f);
	if (bCursorOn)
	{
		// The cursor breathes on the GPU (pulse effect), so blinking never costs a page rebuild.
		WriteText(OutPage[R - 1], Shown.Len(), TEXT("\u2588"), InputColor, 0.6f, ETomeEffect::Pulse, 0.85f);
	}

	// The tail of the log fills rows 2 .. R-4 (R-3 stays blank as breathing room).
	const int32 LogRows = R - 5;
	const int32 Start = FMath::Max(0, Lines.Num() - LogRows);
	for (int32 i = Start; i < Lines.Num(); ++i)
	{
		OutPage[2 + (i - Start)] = Lines[i];
	}
}
