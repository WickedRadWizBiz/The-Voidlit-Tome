"""Command line: play in a terminal, serve the Unreal client, or retrain the Director."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import textwrap
import warnings

from .tome.frames import TomeFrame

RESET = "\x1b[0m"
WIDTH = 88


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _ansi(text: str, color: str, glow: float) -> str:
    r, g, b = _rgb(color)
    bold = "\x1b[1m" if glow >= 0.6 else ""
    return f"{bold}\x1b[38;2;{r};{g};{b}m{text}{RESET}"


def render_ansi(frame: TomeFrame | dict) -> str:
    d = frame.to_dict() if isinstance(frame, TomeFrame) else frame
    out: list[str] = []
    for block in d["blocks"]:
        spans = block["spans"]
        if block["role"] in ("narration", "npc", "title") and len(spans) <= 2:
            text = "".join(s["text"] for s in spans)
            for para in text.split("\n\n"):
                wrapped = textwrap.fill(para, WIDTH)
                out.append(_ansi(wrapped, spans[-1]["color"], spans[-1]["glow"]))
        else:
            out.append("".join(_ansi(s["text"], s["color"], s["glow"]) for s in spans))
        if block["role"] in ("title", "narration"):
            out.append("")
    st = d["status"]
    grit, lumen, chap = st["grit"], st["lumen"], st["chapter"]
    wounds = len([w for w in st["bearer"]["wounds"] if not w.get("stabilized")])
    where = f"  [{chap['index']}/{chap['total']}] {chap['title']}" if chap["total"] else ""
    ring = f" {chap['progress']}/{chap['depth']}" if chap["depth"] else ""
    bar = (f"GRIT {grit[0]}/{grit[1]}  STRESS {st['stress']}  WOUNDS {wounds}  "
           f"LUMEN {'#' * lumen[0]}{'.' * max(0, lumen[1] - lumen[0])}{where}{ring}")
    out.append(_ansi(bar, "#6f7a8f", 0.0))
    return "\n".join(out)


def _load_models(args: argparse.Namespace):
    """Load the local .gguf models named on the command line; None, None when there are none."""
    from .app import load_models
    from .voice.llama_local import LocalModelError

    try:
        return load_models(args.model, args.senses_model, n_ctx=args.ctx, n_gpu_layers=args.gpu_layers)
    except LocalModelError as e:
        print(f"voidlit: {e}", file=sys.stderr)
        return None


def cmd_play(args: argparse.Namespace) -> int:
    from .app import build_engine

    models = _load_models(args)
    if models is None:
        return 2
    engine, llm = build_engine(args.seed, args.cls, args.name, args.place, *models, lower=args.lower,
                               length=args.length, codex=args.content)
    b = engine.bearer
    stats = "  ".join(f"{k.upper()} {v}" for k, v in b.stats().items())
    print(_ansi(f"[{'local llama.cpp' if llm else 'offline voice'}]  {b.name}, the {b.class_name}  {stats}",
                "#6f7a8f", 0.0))
    print(render_ansi(engine.start()))
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text in (":q", "quit", "exit"):
            return 0
        if text.startswith(":save"):
            print(f"saved to {engine.save(text.split(maxsplit=1)[1] if ' ' in text else 'saves/quicksave')}")
            continue
        if text.startswith(":load"):
            engine.load(text.split(maxsplit=1)[1] if " " in text else "saves/quicksave")
            print("loaded.")
            continue
        frame = engine.turn(text)
        print()
        print(render_ansi(frame))
        if args.debug:
            dbg = frame.debug
            print(_ansi(f"  beat={dbg['beat']} probs={dbg['beat_probs']}", "#555c6a", 0.0))
            for e in dbg["events"]:
                print(_ansi(f"  event {e['kind']}: {e['text']}", "#555c6a", 0.0))
            for nid, dec in dbg["decisions"].items():
                print(_ansi(f"  {nid}: plot={dec['dominant_plot']} tactic={dec['tactic']} C={dec['momentum']} "
                            f"R={dec['reward']} events={dec['events']}", "#555c6a", 0.0))
        if frame.over:
            return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import ensure_loopback, run_server

    try:
        ensure_loopback(args.host)
    except ValueError as e:
        print(f"voidlit: {e}", file=sys.stderr)
        return 2
    models = _load_models(args)
    if models is None:
        return 2
    try:
        asyncio.run(run_server(args.host, args.port, *models, save_root=args.saves, debug=args.debug))
    except KeyboardInterrupt:
        pass
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .cortex.director import DEFAULT_WEIGHTS, save_director, train_director

    model, losses = train_director(steps=args.steps, seed=args.seed, log_every=100)
    out = Path(args.out) if args.out else DEFAULT_WEIGHTS
    save_director(model, out)
    print(f"final KL {losses[-1]:.4f}; saved {out}")
    return 0


def cmd_gbnf(args: argparse.Namespace) -> int:
    from .senses.grammar import build_sensory_gbnf

    print(build_sensory_gbnf([t for t in args.targets.split(",") if t]))
    return 0


def _model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None, metavar="FILE.gguf",
                   help="local llama.cpp model for narration and dialogue (in-process; no server, no network)")
    p.add_argument("--senses-model", default=None, metavar="FILE.gguf",
                   help="optional smaller local model for the sensory pass (defaults to --model)")
    p.add_argument("--gpu-layers", type=int, default=0, help="layers to offload to the GPU (0 = CPU only)")
    p.add_argument("--ctx", type=int, default=4096, help="context length")


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore", module="bindsnet")
    p = argparse.ArgumentParser(prog="voidlit", description="The Voidlit Tome: a spiking-neural game master.")
    sub = p.add_subparsers(dest="cmd", required=True)

    play = sub.add_parser("play", help="play in this terminal")
    play.add_argument("--seed", type=int, default=0)
    play.add_argument("--class", dest="cls", default=None,
                      help="breacher, archivist, vessel, hauler, pilgrim or fence (default: the TOME chooses)")
    play.add_argument("--name", default="Bearer")
    play.add_argument("--place", default=None, help="start in a place, e.g. hollow-lantern")
    play.add_argument("--lower", default=None, help="the stat the Opening roll lowers (default: your weakest)")
    play.add_argument("--length", default="short", choices=("short", "long"), help="trial length")
    play.add_argument("--content", default=None, help="a TOME Forge content folder or export bundle")
    _model_args(play)
    play.add_argument("--debug", action="store_true", help="show the brains' decisions each turn")
    play.set_defaults(fn=cmd_play)

    srv = sub.add_parser("serve", help="serve the Unreal client over WebSocket")
    srv.add_argument("--host", default="127.0.0.1", help="loopback only (127.0.0.1, ::1 or localhost)")
    srv.add_argument("--port", type=int, default=8765)
    _model_args(srv)
    srv.add_argument("--saves", default="saves")
    srv.add_argument("--debug", action="store_true", help="include brain internals in every frame")
    srv.set_defaults(fn=cmd_serve)

    tr = sub.add_parser("train-director", help="retrain the Director SNN from the pacing teacher")
    tr.add_argument("--steps", type=int, default=800)
    tr.add_argument("--seed", type=int, default=7)
    tr.add_argument("--out", default=None)
    tr.set_defaults(fn=cmd_train)

    g = sub.add_parser("gbnf", help="print the sensory grammar")
    g.add_argument("--targets", default="")
    g.set_defaults(fn=cmd_gbnf)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.cmd == "serve" else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
