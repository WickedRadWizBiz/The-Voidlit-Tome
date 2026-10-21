"""WebSocket server: the Tome's mind, serving its voxel body (Unreal) over localhost.

Protocol (JSON text messages; see docs/PROTOCOL.md):

client -> server
    {"type": "hello", "protocol": 2, "options": {"seed": 7, "background": "medic", "name": "Ash"}}
    {"type": "input", "text": "I pry the door open"}
    {"type": "command", "name": "new" | "save" | "load", "args": {...}}
    {"type": "ping"}

server -> client
    {"type": "welcome", ...}   once, after hello
    {"type": "thinking"}       as soon as input arrives, so the Tome can ripple while it thinks
    {"type": "frame", ...}     the result (see voidlit.tome.frames)
    {"type": "saved" | "loaded" | "pong" | "error", ...}
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
from pathlib import Path

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from .app import build_engine
from .dev import require_dev
from .tome.frames import PROTOCOL_VERSION
from .voice.llama_local import ChatModel

log = logging.getLogger(__name__)
MAX_INPUT = 2000


class TomeSession:
    def __init__(self, ws: ServerConnection, voice_llm: ChatModel | None, senses_llm: ChatModel | None,
                 save_root: Path, debug: bool) -> None:
        self.ws = ws
        self.voice_llm = voice_llm
        self.senses_llm = senses_llm
        self.save_root = save_root
        self.debug = debug
        self.engine = None
        self.lock = asyncio.Lock()

    async def send(self, msg: dict) -> None:
        await self.ws.send(json.dumps(msg))

    async def send_frame(self, frame) -> None:
        await self.send(frame.to_dict(include_debug=self.debug))

    async def new_game(self, options: dict) -> None:
        engine, llm = await asyncio.to_thread(
            build_engine, int(options.get("seed", 0)), options.get("background"),
            str(options.get("name", "Wanderer"))[:32], options.get("scenario", "hollow_lantern"),
            self.voice_llm, self.senses_llm,
        )
        self.engine = engine
        await self.send({
            "type": "welcome", "protocol": PROTOCOL_VERSION, "scenario": engine.scenario.title,
            "backgrounds": {k: v.name for k, v in engine.scenario.backgrounds.items()},
            "character": engine.state.character.sheet(), "llm": llm,
        })
        await self.send_frame(await asyncio.to_thread(engine.start))

    async def handle(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "ping":
            await self.send({"type": "pong"})
        elif kind == "hello":
            await self.new_game(msg.get("options", {}))
        elif self.engine is None:
            await self.send({"type": "error", "message": "say hello first"})
        elif kind == "input":
            text = str(msg.get("text", "")).strip()[:MAX_INPUT]
            if not text:
                return
            await self.send({"type": "thinking"})
            async with self.lock:
                frame = await asyncio.to_thread(self.engine.turn, text)
            await self.send_frame(frame)
        elif kind == "command":
            await self.command(msg.get("name", ""), msg.get("args", {}))
        else:
            await self.send({"type": "error", "message": f"unknown message type {kind!r}"})

    async def command(self, name: str, args: dict) -> None:
        slot = "".join(c for c in str(args.get("slot", "quicksave")) if c.isalnum() or c in "-_")[:40] or "quicksave"
        if name == "new":
            await self.new_game(args)
        elif name == "save":
            path = await asyncio.to_thread(self.engine.save, self.save_root / slot)
            await self.send({"type": "saved", "slot": slot, "path": str(path)})
        elif name == "load":
            try:
                await asyncio.to_thread(self.engine.load, self.save_root / slot)
            except (OSError, ValueError, KeyError) as e:
                await self.send({"type": "error", "message": f"load failed: {e}"})
                return
            await self.send({"type": "loaded", "slot": slot})
        else:
            await self.send({"type": "error", "message": f"unknown command {name!r}"})


def ensure_loopback(host: str) -> None:
    """The Tome serves only this machine: refuse any address that is not loopback."""
    if host == "localhost":
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise ValueError(f"refusing to listen on {host!r}: the Tome only serves this machine (use 127.0.0.1)")


async def run_server(host: str = "127.0.0.1", port: int = 8765, voice_llm: ChatModel | None = None,
                     senses_llm: ChatModel | None = None, save_root: str | Path = "saves", debug: bool = False,
                     ready: asyncio.Event | None = None) -> None:
    ensure_loopback(host)
    if debug:
        require_dev("--debug")
    root = Path(save_root)

    async def handler(ws: ServerConnection) -> None:
        session = TomeSession(ws, voice_llm, senses_llm, root, debug)
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    if not isinstance(msg, dict):
                        raise ValueError("message must be a JSON object")
                except ValueError as e:
                    await session.send({"type": "error", "message": f"bad message: {e}"})
                    continue
                await session.handle(msg)
        except ConnectionClosed:
            pass

    async with serve(handler, host, port, max_size=2**20) as server:
        log.info("The Tome is listening on ws://%s:%d", host, port)
        if ready is not None:
            ready.set()
        await server.serve_forever()
