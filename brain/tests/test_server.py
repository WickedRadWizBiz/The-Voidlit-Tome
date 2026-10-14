import asyncio
import json
import socket

from websockets.asyncio.client import connect

from voidlit.server import run_server


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _session(tmp_path):
    port = free_port()
    ready = asyncio.Event()
    task = asyncio.create_task(run_server("127.0.0.1", port, save_root=tmp_path, debug=True, ready=ready))
    await asyncio.wait_for(ready.wait(), 10)
    try:
        async with connect(f"ws://127.0.0.1:{port}") as ws:
            async def recv():
                return json.loads(await asyncio.wait_for(ws.recv(), 60))

            await ws.send(json.dumps({"type": "input", "text": "hi"}))
            assert (await recv())["type"] == "error"

            await ws.send(json.dumps({"type": "hello", "options": {"seed": 3, "class": "hauler", "name": "Kit",
                                                                "place": "hollow_lantern"}}))
            welcome = await recv()
            assert welcome["type"] == "welcome" and welcome["llm"] is False
            assert welcome["bearer"]["class_name"] == "Hauler" and "breacher" in welcome["classes"]
            assert welcome["protocol"] == 2 and "hollow_lantern" in welcome["places"]
            start = await recv()
            assert start["type"] == "frame" and start["blocks"][0]["role"] == "title"
            assert start["status"]["pending"]["type"] == "draw"

            await ws.send(json.dumps({"type": "input", "text": "I touch the first one"}))
            assert (await recv())["type"] == "thinking"
            frame = await recv()
            assert frame["type"] == "frame" and frame["seq"] == start["seq"] + 1 and "debug" in frame

            await ws.send(json.dumps({"type": "command", "name": "save", "args": {"slot": "../evil"}}))
            saved = await recv()
            assert saved["type"] == "saved" and saved["slot"] == "evil"
            await ws.send(json.dumps({"type": "command", "name": "load", "args": {"slot": "evil"}}))
            assert (await recv())["type"] == "loaded"

            await ws.send("not json")
            assert (await recv())["type"] == "error"
            await ws.send(json.dumps({"type": "ping"}))
            assert (await recv())["type"] == "pong"
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def test_websocket_protocol(tmp_path):
    asyncio.run(_session(tmp_path))
