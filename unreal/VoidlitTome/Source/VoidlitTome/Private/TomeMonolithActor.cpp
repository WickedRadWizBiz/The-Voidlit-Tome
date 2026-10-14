#include "TomeMonolithActor.h"

#include "Components/StaticMeshComponent.h"
#include "Engine/Engine.h"
#include "Engine/GameViewportClient.h"
#include "Engine/StaticMesh.h"
#include "Framework/Application/SlateApplication.h"
#include "GameFramework/PlayerController.h"
#include "Kismet/GameplayStatics.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "TomeVoxelSurfaceComponent.h"
#include "UObject/ConstructorHelpers.h"
#include "VoidlitTomeModule.h"
#include "Widgets/Input/SEditableText.h"
#include "Widgets/Layout/SBox.h"

ATomeMonolithActor::ATomeMonolithActor()
{
	PrimaryActorTick.bCanEverTick = true;

	Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	SetRootComponent(Root);

	Slab = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Slab"));
	Slab->SetupAttachment(Root);
	static ConstructorHelpers::FObjectFinder<UStaticMesh> Cube(TEXT("/Engine/BasicShapes/Cube.Cube"));
	if (Cube.Succeeded())
	{
		Slab->SetStaticMesh(Cube.Object);
	}

	Surface = CreateDefaultSubobject<UTomeVoxelSurfaceComponent>(TEXT("Surface"));
	Surface->SetupAttachment(Root);
}

void ATomeMonolithActor::LayoutSlab()
{
	const float Width = Surface->Columns * Surface->VoxelPitchCm;
	const float Height = Surface->Rows * Surface->VoxelPitchCm;
	// Origin at the bottom centre of the slab; the face looks along +X.
	Slab->SetRelativeScale3D(FVector(SlabThicknessCm, Width, Height) / 100.f);
	Slab->SetRelativeLocation(FVector(0.f, 0.f, 0.5f * Height));
	// Voxel tops sit 0.05 mm proud of the slab's face so the face itself never z-fights them.
	Surface->SetRelativeLocation(FVector(0.5f * SlabThicknessCm + 0.005f, 0.f, 0.5f * Height));
}

void ATomeMonolithActor::OnConstruction(const FTransform& Transform)
{
	Super::OnConstruction(Transform);
	LayoutSlab();
}

void ATomeMonolithActor::BeginPlay()
{
	Super::BeginPlay();
	LayoutSlab();

	// The slab shares the voxels' matte black so flush voxels disappear into it.
	if (UMaterialInterface* Matte = LoadObject<UMaterialInterface>(nullptr, TEXT("/VoidlitTome/M_TomeSlab.M_TomeSlab")))
	{
		Slab->SetMaterial(0, Matte);
	}

	Typesetter.SetColumns(Surface->GetTextColumns());
	Refresh();

	if (bSpawnBrain)
	{
		SpawnBrain();
	}
	Brain = NewObject<UTomeBrainClient>(this);
	Brain->OnFrame.AddDynamic(this, &ATomeMonolithActor::HandleFrame);
	Brain->OnThinking.AddDynamic(this, &ATomeMonolithActor::HandleThinking);
	Brain->OnConnectionChanged.AddDynamic(this, &ATomeMonolithActor::HandleConnection);
	Brain->Connect(BrainUrl, SessionOptions);

	if (bCaptureKeyboard)
	{
		CreateInputCatcher();
	}
}

void ATomeMonolithActor::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	RemoveInputCatcher();
	if (Brain)
	{
		Brain->Disconnect();
	}
	if (BrainProcess.IsValid())
	{
		FPlatformProcess::TerminateProc(BrainProcess, /*KillTree*/ true);
		FPlatformProcess::CloseProc(BrainProcess);
	}
	Super::EndPlay(EndPlayReason);
}

void ATomeMonolithActor::SpawnBrain()
{
	const TCHAR* WorkingDir = BrainWorkingDirectory.IsEmpty() ? nullptr : *BrainWorkingDirectory;
	BrainProcess = FPlatformProcess::CreateProc(*PythonExecutable, *BrainArguments, /*bLaunchDetached*/ false,
		/*bLaunchHidden*/ true, /*bLaunchReallyHidden*/ true, nullptr, 0, WorkingDir, nullptr);
	if (!BrainProcess.IsValid())
	{
		UE_LOG(LogVoidlitTome, Error, TEXT("Could not start the Tome brain: %s %s"), *PythonExecutable,
			*BrainArguments);
	}
}

void ATomeMonolithActor::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	if (Brain)
	{
		Brain->Tick(DeltaSeconds);
	}

	// Keep the keyboard on the Tome.
	if (InputWidget.IsValid() && FSlateApplication::IsInitialized() && !InputWidget->HasKeyboardFocus())
	{
		FSlateApplication::Get().SetKeyboardFocus(InputWidget, EFocusCause::SetDirectly);
	}
}

void ATomeMonolithActor::Refresh(int32 NewTextFromSerial)
{
	TArray<FTomeLine> Page;
	Typesetter.ComposePage(Surface->GetTextRows(), LastStatus, InputText, /*bCursorOn*/ true, Page);
	Surface->ShowPage(Page, NewTextFromSerial);
}

void ATomeMonolithActor::HandleFrame(const FTomeFrame& Frame)
{
	LastStatus = Frame.Status;
	const int32 FirstNew = Typesetter.AppendFrame(Frame);
	Surface->SetThinking(false);
	Surface->SetMood(Frame.Mood.Lumen, Frame.Mood.bDark, Frame.Mood.bHeartbeat);
	for (const FTomeEffectCue& Cue : Frame.Effects)
	{
		Surface->PlayCue(Cue.Kind, Cue.Strength);
	}
	for (const FTomeBlock& Block : Frame.Blocks)
	{
		if (Block.Role == TEXT("title"))
		{
			Surface->PlayCue(TEXT("ripple"), 0.6f);
			break;
		}
	}
	Refresh(FirstNew);
}

void ATomeMonolithActor::HandleThinking()
{
	Surface->SetThinking(true);
}

void ATomeMonolithActor::HandleConnection(bool bInConnected)
{
	bConnected = bInConnected;
	if (!bConnected)
	{
		Surface->SetThinking(true);  // the surface breathes while it waits for its mind
	}
}

void ATomeMonolithActor::SetInputText(const FString& Text)
{
	InputText = Text.Left(MaxInputChars);
	Refresh();
}

void ATomeMonolithActor::SubmitInput(const FString& Text)
{
	const FString Line = Text.TrimStartAndEnd();
	SetInputText(FString());
	if (Line.IsEmpty() || !Brain)
	{
		return;
	}
	if (Line.StartsWith(TEXT(":")))
	{
		FString Command, Slot;
		if (!Line.Mid(1).Split(TEXT(" "), &Command, &Slot))
		{
			Command = Line.Mid(1);
		}
		Brain->SendCommand(Command.ToLower(), Slot.TrimStartAndEnd());
		if (Command.ToLower() == TEXT("new"))
		{
			Typesetter.Clear();
			Refresh();
		}
		return;
	}
	Brain->SendInput(Line.Left(MaxInputChars));
}

void ATomeMonolithActor::CreateInputCatcher()
{
	if (!GEngine || !GEngine->GameViewport || !FSlateApplication::IsInitialized())
	{
		return;
	}
	TWeakObjectPtr<ATomeMonolithActor> WeakThis(this);
	SAssignNew(InputWidget, SEditableText)
		.ClearKeyboardFocusOnCommit(false)
		.SelectAllTextWhenFocused(false)
		.OnTextChanged_Lambda([WeakThis](const FText& NewText)
		{
			if (WeakThis.IsValid())
			{
				WeakThis->SetInputText(NewText.ToString());
			}
		})
		.OnTextCommitted_Lambda([WeakThis](const FText& Committed, ETextCommit::Type How)
		{
			if (How == ETextCommit::OnEnter && WeakThis.IsValid())
			{
				WeakThis->SubmitInput(Committed.ToString());
				if (WeakThis->InputWidget.IsValid())
				{
					WeakThis->InputWidget->SetText(FText::GetEmpty());
				}
			}
		});
	InputWidget->SetRenderOpacity(0.f);  // invisible: the voxels are the only text on screen
	InputRoot = SNew(SBox).WidthOverride(1.f).HeightOverride(1.f)[InputWidget.ToSharedRef()];
	GEngine->GameViewport->AddViewportWidgetContent(InputRoot.ToSharedRef(), /*ZOrder*/ 10);

	if (APlayerController* PC = UGameplayStatics::GetPlayerController(this, 0))
	{
		FInputModeUIOnly Mode;
		Mode.SetWidgetToFocus(InputWidget);
		PC->SetInputMode(Mode);
		PC->SetShowMouseCursor(false);
	}
	FSlateApplication::Get().SetKeyboardFocus(InputWidget, EFocusCause::SetDirectly);
}

void ATomeMonolithActor::RemoveInputCatcher()
{
	if (InputRoot.IsValid() && GEngine && GEngine->GameViewport)
	{
		GEngine->GameViewport->RemoveViewportWidgetContent(InputRoot.ToSharedRef());
	}
	InputRoot.Reset();
	InputWidget.Reset();
}
