#include "TomeBrainClient.h"

#include "IWebSocket.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "VoidlitTomeModule.h"
#include "WebSocketsModule.h"

static constexpr int32 TomeProtocolVersion = 2;

void UTomeBrainClient::Connect(const FString& Url, const FTomeSessionOptions& Options)
{
	ServerUrl = Url;
	SessionOptions = Options;
	bWantConnection = true;
	Backoff = 1.f;
	Open();
}

void UTomeBrainClient::Open()
{
	if (Socket.IsValid())
	{
		Socket->OnConnected().RemoveAll(this);
		Socket->OnConnectionError().RemoveAll(this);
		Socket->OnClosed().RemoveAll(this);
		Socket->OnMessage().RemoveAll(this);
		Socket->Close();
		Socket.Reset();
	}
	bHelloSent = false;
	ReconnectIn = -1.f;
	FWebSocketsModule& WebSockets = FModuleManager::LoadModuleChecked<FWebSocketsModule>(TEXT("WebSockets"));
	Socket = WebSockets.CreateWebSocket(ServerUrl);
	Socket->OnConnected().AddUObject(this, &UTomeBrainClient::HandleConnected);
	Socket->OnConnectionError().AddUObject(this, &UTomeBrainClient::HandleConnectionError);
	Socket->OnClosed().AddUObject(this, &UTomeBrainClient::HandleClosed);
	Socket->OnMessage().AddUObject(this, &UTomeBrainClient::HandleMessage);
	Socket->Connect();
}

void UTomeBrainClient::Disconnect()
{
	bWantConnection = false;
	if (Socket.IsValid())
	{
		Socket->Close();
	}
}

bool UTomeBrainClient::IsConnected() const
{
	return Socket.IsValid() && Socket->IsConnected();
}

void UTomeBrainClient::BeginDestroy()
{
	bWantConnection = false;
	if (Socket.IsValid())
	{
		Socket->OnConnected().RemoveAll(this);
		Socket->OnConnectionError().RemoveAll(this);
		Socket->OnClosed().RemoveAll(this);
		Socket->OnMessage().RemoveAll(this);
		Socket->Close();
		Socket.Reset();
	}
	Super::BeginDestroy();
}

void UTomeBrainClient::Tick(float DeltaTime)
{
	if (ReconnectIn >= 0.f && bWantConnection)
	{
		ReconnectIn -= DeltaTime;
		if (ReconnectIn < 0.f)
		{
			Open();
		}
	}
}

void UTomeBrainClient::SendJson(const TSharedRef<FJsonObject>& Obj)
{
	if (!IsConnected())
	{
		return;
	}
	FString Out;
	const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Out);
	FJsonSerializer::Serialize(Obj, Writer);
	Socket->Send(Out);
}

void UTomeBrainClient::SendInput(const FString& Text)
{
	const TSharedRef<FJsonObject> Msg = MakeShared<FJsonObject>();
	Msg->SetStringField(TEXT("type"), TEXT("input"));
	Msg->SetStringField(TEXT("text"), Text);
	SendJson(Msg);
}

void UTomeBrainClient::SendCommand(const FString& Name, const FString& Slot)
{
	const TSharedRef<FJsonObject> Args = MakeShared<FJsonObject>();
	if (!Slot.IsEmpty())
	{
		Args->SetStringField(TEXT("slot"), Slot);
	}
	if (Name == TEXT("new"))
	{
		Args->SetNumberField(TEXT("seed"), SessionOptions.Seed);
		Args->SetStringField(TEXT("name"), SessionOptions.PlayerName);
		if (!SessionOptions.Background.IsEmpty())
		{
			Args->SetStringField(TEXT("background"), SessionOptions.Background);
		}
	}
	const TSharedRef<FJsonObject> Msg = MakeShared<FJsonObject>();
	Msg->SetStringField(TEXT("type"), TEXT("command"));
	Msg->SetStringField(TEXT("name"), Name);
	Msg->SetObjectField(TEXT("args"), Args);
	SendJson(Msg);
}

void UTomeBrainClient::HandleConnected()
{
	Backoff = 1.f;
	OnConnectionChanged.Broadcast(true);
	if (!bHelloSent)
	{
		const TSharedRef<FJsonObject> Options = MakeShared<FJsonObject>();
		Options->SetNumberField(TEXT("seed"), SessionOptions.Seed);
		Options->SetStringField(TEXT("name"), SessionOptions.PlayerName);
		if (!SessionOptions.Background.IsEmpty())
		{
			Options->SetStringField(TEXT("background"), SessionOptions.Background);
		}
		const TSharedRef<FJsonObject> Msg = MakeShared<FJsonObject>();
		Msg->SetStringField(TEXT("type"), TEXT("hello"));
		Msg->SetNumberField(TEXT("protocol"), TomeProtocolVersion);
		Msg->SetObjectField(TEXT("options"), Options);
		SendJson(Msg);
		bHelloSent = true;
	}
}

void UTomeBrainClient::HandleConnectionError(const FString& Error)
{
	UE_LOG(LogVoidlitTome, Warning, TEXT("Tome brain unreachable at %s (%s); retrying in %.0fs"), *ServerUrl, *Error,
		Backoff);
	OnConnectionChanged.Broadcast(false);
	ReconnectIn = Backoff;
	Backoff = FMath::Min(Backoff * 2.f, 16.f);
}

void UTomeBrainClient::HandleClosed(int32 StatusCode, const FString& Reason, bool bWasClean)
{
	OnConnectionChanged.Broadcast(false);
	if (bWantConnection)
	{
		ReconnectIn = Backoff;
		Backoff = FMath::Min(Backoff * 2.f, 16.f);
	}
}

void UTomeBrainClient::HandleMessage(const FString& Message)
{
	TSharedPtr<FJsonObject> Obj;
	const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Message);
	if (!FJsonSerializer::Deserialize(Reader, Obj) || !Obj.IsValid())
	{
		UE_LOG(LogVoidlitTome, Warning, TEXT("Unparseable message from the Tome brain"));
		return;
	}
	const FString Type = Obj->GetStringField(TEXT("type"));
	if (Type == TEXT("frame"))
	{
		FTomeFrame Frame;
		if (TomeJson::ParseFrame(Obj, Frame))
		{
			OnFrame.Broadcast(Frame);
		}
	}
	else if (Type == TEXT("thinking"))
	{
		OnThinking.Broadcast();
	}
	else if (Type == TEXT("welcome"))
	{
		bool bLlm = false;
		Obj->TryGetBoolField(TEXT("llm"), bLlm);
		OnWelcome.Broadcast(Obj->GetStringField(TEXT("scenario")), bLlm);
	}
	else if (Type == TEXT("error"))
	{
		const FString Text = Obj->GetStringField(TEXT("message"));
		UE_LOG(LogVoidlitTome, Warning, TEXT("Tome brain error: %s"), *Text);
		OnError.Broadcast(Text);
	}
}
