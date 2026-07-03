"""Command-line entrypoint for the chess engine pipeline.

Examples:
    python -m cli label --generate 2000 --depth 12
    python -m cli supervised --epochs 10
    python -m cli selfplay --iterations 20 --games-per-iter 10
    python -m cli evaluate --checkpoint models/supervised.pt --games 20
    python -m cli serve --port 8000
"""
from __future__ import annotations

import argparse
import os
from typing import List

from engine.config import Config


def cmd_generate(args: argparse.Namespace) -> None:
    from data.generate import generate_positions

    fens = generate_positions(args.n, seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(fens))
    print(f"Wrote {len(fens)} FENs to {args.out}")


def cmd_label(args: argparse.Namespace) -> None:
    import os as _os
    import time as _time

    from data.generate import generate_positions
    from data.label import label_positions
    from train.progress import ProgressLogger

    config = Config()
    config.ensure_dirs()
    # Create the log file up front so the UI has something to stream immediately.
    logger = ProgressLogger(_os.path.join(config.logs_dir, "label.jsonl"))

    if args.generate:
        logger.log({"event": "generate", "status": "start", "n": args.generate})
        t0 = _time.time()
        gen_state = {"last": 0.0}

        def _gen_progress(done: int, total: int) -> None:
            now = _time.time()
            if now - gen_state["last"] >= 1.5 or done >= total:
                rate = done / (now - t0) if now > t0 else 0.0
                logger.log(
                    {
                        "event": "generate",
                        "status": "progress",
                        "n": done,
                        "target": total,
                        "rate_pos_per_sec": round(rate),
                    }
                )
                gen_state["last"] = now

        fens = generate_positions(args.generate, seed=args.seed, progress=_gen_progress)
        logger.log(
            {
                "event": "generate",
                "status": "done",
                "n": len(fens),
                "elapsed_s": round(_time.time() - t0, 1),
            }
        )
    elif args.fens:
        with open(args.fens) as f:
            fens = [line.strip() for line in f if line.strip()]
        logger.log({"event": "generate", "status": "loaded", "n": len(fens)})
    else:
        raise SystemExit("Provide --generate N or --fens FILE")
    shards = label_positions(
        fens,
        config=config,
        shard_size=args.shard_size,
        depth=args.depth,
        movetime=args.movetime,
        multipv=args.multipv,
        threads=args.threads,
        workers=args.workers,
        progress=logger,
    )
    print(f"Labeled {len(fens)} positions into {len(shards)} new shard(s).")


def cmd_supervised(args: argparse.Namespace) -> None:
    from train.supervised import train_supervised

    config = Config()
    # Network size overrides (only applied when training a fresh model; when
    # resuming, the architecture is taken from the checkpoint).
    if args.blocks is not None:
        config.model.num_blocks = args.blocks
    if args.channels is not None:
        config.model.channels = args.channels
    if args.value_hidden is not None:
        config.model.value_hidden = args.value_hidden
    path = train_supervised(
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        limit=args.limit,
        resume=args.resume,
        out_name=args.out,
        num_workers=args.workers,
    )
    print(f"Saved supervised checkpoint to {path}")


def cmd_selfplay(args: argparse.Namespace) -> None:
    from train.train_loop import train_selfplay

    config = Config()
    path = train_selfplay(
        config=config,
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        sims=args.sims,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        init_checkpoint=args.init,
        out_name=args.out,
        sf_value_weight=args.sf_value_weight,
        workers=args.workers,
        selfplay_device=args.selfplay_device,
        eval_every=args.eval_every,
    )
    print(f"Saved self-play checkpoint to {path}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from engine.model import build_model, load_checkpoint
    from engine.player import EnginePlayer
    from train.evaluate import estimate_elo

    config = Config()
    if args.checkpoint and os.path.exists(args.checkpoint):
        model, _ = load_checkpoint(args.checkpoint, device=config.device)
    else:
        print("No checkpoint found; evaluating a randomly-initialized net.")
        model = build_model(config.model, device=config.device)
    model.eval()
    player = EnginePlayer(model, config)
    result = estimate_elo(
        player,
        config,
        games=args.games,
        sims=args.sims,
        skill_level=args.skill,
        movetime=args.movetime,
    )
    print(result)


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    os.environ.setdefault("CHESSAI_CHECKPOINT", args.checkpoint or "")
    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=args.reload)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AlphaZero-style chess engine pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Generate positions (FENs)")
    g.add_argument("--n", type=int, default=2000)
    g.add_argument("--out", default="data_store/fens.txt")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_generate)

    la = sub.add_parser("label", help="Label positions with Stockfish")
    la.add_argument("--generate", type=int, default=0, help="Generate N positions then label")
    la.add_argument("--fens", default=None, help="Path to a file of FENs")
    la.add_argument("--shard-size", type=int, default=2000)
    la.add_argument("--depth", type=int, default=12)
    la.add_argument("--movetime", type=float, default=None)
    la.add_argument("--multipv", type=int, default=4)
    la.add_argument("--threads", type=int, default=1, help="Stockfish threads per worker")
    la.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Parallel Stockfish worker processes (defaults to half your cores)",
    )
    la.add_argument("--seed", type=int, default=0)
    la.set_defaults(func=cmd_label)

    s = sub.add_parser("supervised", help="Supervised bootstrap training")
    s.add_argument("--epochs", type=int, default=10)
    s.add_argument("--batch-size", type=int, default=256)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--resume", default=None)
    s.add_argument("--out", default="supervised.pt")
    s.add_argument("--blocks", type=int, default=None, help="Residual blocks (network depth)")
    s.add_argument("--channels", type=int, default=None, help="Conv channels (network width)")
    s.add_argument("--value-hidden", type=int, default=None, help="Value head hidden units")
    s.add_argument(
        "--workers", type=int, default=0, help="DataLoader worker processes (use on GPU, e.g. 8)"
    )
    s.set_defaults(func=cmd_supervised)

    sp = sub.add_parser("selfplay", help="Self-play refinement loop")
    sp.add_argument("--iterations", type=int, default=20)
    sp.add_argument("--games-per-iter", type=int, default=10)
    sp.add_argument("--sims", type=int, default=100)
    sp.add_argument("--train-steps", type=int, default=200)
    sp.add_argument("--batch-size", type=int, default=128)
    sp.add_argument("--init", default="models/supervised.pt")
    sp.add_argument("--out", default="selfplay.pt")
    sp.add_argument("--sf-value-weight", type=float, default=0.0)
    sp.add_argument(
        "--workers", type=int, default=1, help="Parallel self-play worker processes (e.g. 12)"
    )
    sp.add_argument(
        "--selfplay-device",
        default=None,
        help="Device for self-play workers (default: cpu when workers>1, else the training device)",
    )
    sp.add_argument("--eval-every", type=int, default=5)
    sp.set_defaults(func=cmd_selfplay)

    e = sub.add_parser("evaluate", help="Estimate Elo vs Stockfish")
    e.add_argument("--checkpoint", default="models/best.pt")
    e.add_argument("--games", type=int, default=20)
    e.add_argument("--sims", type=int, default=80)
    e.add_argument("--skill", type=int, default=3)
    e.add_argument("--movetime", type=float, default=0.05)
    e.set_defaults(func=cmd_evaluate)

    sv = sub.add_parser("serve", help="Run the web app")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--checkpoint", default=None)
    sv.add_argument("--reload", action="store_true")
    sv.set_defaults(func=cmd_serve)

    return p


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
