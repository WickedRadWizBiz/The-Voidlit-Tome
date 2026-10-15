#pragma once

#include "CoreMinimal.h"
#include "TomeTypes.h"
#include "UObject/Object.h"

#include "TomeBrainClient.generated.h"

class IWebSocket;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FTomeFrameEvent, const FTomeFrame&, Frame);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FTomeThinkingEvent);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FTomeWelcomeEvent, const FString&, BearerClass, bool, bLanguageModelOnline);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FTomeConnectionEvent, bool, bConnected);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FTomeErrorEvent, const FString&, Message);

/** Options sent with "hello" (docs/PROTOCOL.md). */
USTRUCT(BlueprintType)
struct VOIDLITTOME_API FTomeSessionOptions
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Tome")
	int32 Seed = 0;

	/** breacher | archivist | vessel | hauler | pilgrim | fence; empty = the TOME chooses */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Tome")
	FString BearerClass;

	/** a place pack such as hollow_lantern; empty = a Threshold roll */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Tome")
	FString Place;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Tome")
	FString PlayerName = TEXT("Bearer");
};

/** WebSocket link to the Tome's mind (`python -m voidlit serve`). Reconnects with backoff. */
UCLASS(BlueprintType)
class VOIDLITTOME_API UTomeBrainClient : public UObject
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintCallable, Category = "Tome")
	void Connect(const FString& Url, const FTomeSessionOptions& Options);

	UFUNCTION(BlueprintCallable, Category = "Tome")
	void Disconnect();

	UFUNCTION(BlueprintPure, Category = "Tome")
	bool IsConnected() const;

	UFUNCTION(BlueprintCallable, Category = "Tome")
	void SendInput(const FString& Text);

	/** new | save | load, with an optional slot. */
	UFUNCTION(BlueprintCallable, Category = "Tome")
	void SendCommand(const FString& Name, const FString& Slot);

	/** Drive reconnection; the owning actor calls this every tick. */
	void Tick(float DeltaTime);

	UPROPERTY(BlueprintAssignable, Category = "Tome")
	FTomeFrameEvent OnFrame;

	UPROPERTY(BlueprintAssignable, Category = "Tome")
	FTomeThinkingEvent OnThinking;

	UPROPERTY(BlueprintAssignable, Category = "Tome")
	FTomeWelcomeEvent OnWelcome;

	UPROPERTY(BlueprintAssignable, Category = "Tome")
	FTomeConnectionEvent OnConnectionChanged;

	UPROPERTY(BlueprintAssignable, Category = "Tome")
	FTomeErrorEvent OnError;

	virtual void BeginDestroy() override;

private:
	void Open();
	void SendJson(const TSharedRef<FJsonObject>& Obj);
	void HandleConnected();
	void HandleConnectionError(const FString& Error);
	void HandleClosed(int32 StatusCode, const FString& Reason, bool bWasClean);
	void HandleMessage(const FString& Message);

	TSharedPtr<IWebSocket> Socket;
	FString ServerUrl;
	FTomeSessionOptions SessionOptions;
	bool bWantConnection = false;
	bool bHelloSent = false;
	float ReconnectIn = -1.f;
	float Backoff = 1.f;
};
