# voidlit: the Voidlit Tome's mind

The Python sidecar for [The Voidlit Tome](../README.md): rules engine, BindsNET NPC brains, the SpikingJelly
Director, the llama.cpp transducer and voice, and the WebSocket server the Unreal client talks to.

```bash
pip install -e ".[dev]"
python -m voidlit play            # terminal play, offline
python -m voidlit serve           # ws://127.0.0.1:8765 for Unreal
python -m voidlit train-director  # retrain the pacing SNN (~30 s on CPU)
python -m pytest -q
```

See [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) for how the pieces fit.
