# voidlit: the Voidlit Tome's mind

The Python sidecar for [The Voidlit Tome](../README.md): rules engine, BindsNET NPC brains, the SpikingJelly
Director, the local llama.cpp transducer and voice (in-process, `.gguf` files only), and the loopback WebSocket
server the Unreal client talks to.

```bash
pip install -e ".[dev]"
python -m voidlit play            # terminal play, offline
CMAKE_ARGS="-DLLAMA_CURL=OFF" pip install -e ".[llm]"
python -m voidlit play --model /path/to/model.gguf   # local llama.cpp, in-process
python -m voidlit serve           # ws://127.0.0.1:8765 for Unreal
python -m voidlit train-director  # retrain the pacing SNN (~30 s on CPU)
python -m pytest -q
```

See [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) for how the pieces fit.
