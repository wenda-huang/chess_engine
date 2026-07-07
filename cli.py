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
        lr=args.lr,
        init_checkpoint=args.init,
        out_name=args.out,
        buffer_capacity=args.buffer_capacity,
        sf_value_weight=args.sf_value_weight,
        workers=args.workers,
        selfplay_device=args.selfplay_device,
        sup_fraction=args.sup_fraction,
        arena_every=args.arena_every,
        arena_games=args.arena_games,
        arena_sims=args.arena_sims,
        arena_opening_pool=args.arena_opening_pool,
        arena_book_fraction=args.arena_book_fraction,
        arena_opening_min=args.arena_opening_min,
        arena_opening_max=args.arena_opening_max,
        arena_temperature=args.arena_temperature,
        arena_temp_moves=args.arena_temp_moves,
        gate_threshold=args.gate_threshold,
        gate_min_games=args.gate_min_games,
        gate_require_significance=not args.no_gate_significance,
        temperature_moves=args.temperature_moves,
        resign=not args.no_resign,
        resign_threshold=args.resign_threshold,
        resign_streak=args.resign_streak,
        complete_fraction=args.complete_fraction,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        eval_skill=args.eval_skill,
        eval_sims=args.eval_sims,
    )
    print(f"Saved self-play checkpoint to {path}")


def cmd_probe(args: argparse.Namespace) -> None:
    """Check whether the value head has collapsed to 'always ~0'.

    Samples labeled positions, runs the net, and compares its value predictions
    against the Stockfish value targets. A healthy net has a wide prediction
    spread that correlates with the targets; a collapsed one predicts ~0
    everywhere (near-zero std, near-zero correlation).
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    from data.dataset import StockfishDataset
    from engine.model import load_checkpoint

    config = Config()
    model, _ = load_checkpoint(args.checkpoint, device=config.device)
    model.eval()

    ds = StockfishDataset(config.data_dir)
    if len(ds) == 0:
        raise SystemExit("No labeled data found to probe against.")
    n = min(args.n, len(ds))
    idxs = np.random.RandomState(0).randint(0, len(ds), size=n)

    planes, policies, targets = [], [], []
    for i in idxs:
        p, pol, v = ds[int(i)]
        planes.append(p)
        policies.append(pol)
        targets.append(float(v))
    x = torch.stack(planes).to(config.device)
    tgt_pol = torch.stack(policies).to(config.device)
    with torch.no_grad():
        logits, value = model(x)

    # ---- Value head ----
    preds = value.detach().cpu().numpy().reshape(-1)
    targs = np.asarray(targets, dtype=np.float64)
    pred_std = float(preds.std())
    frac_near_zero = float(np.mean(np.abs(preds) < 0.1))
    v_corr = float(np.corrcoef(preds, targs)[0, 1]) if pred_std > 1e-6 else 0.0
    v_collapsed = pred_std < 0.1 or abs(v_corr) < 0.2

    # ---- Policy head (drift vs Stockfish targets) ----
    logp = F.log_softmax(logits, dim=1)
    probs = logp.exp()
    policy_ce = float(-(tgt_pol * logp).sum(dim=1).mean().item())
    mask = tgt_pol.sum(dim=1) > 0
    top1_agree = float(
        (logits.argmax(dim=1)[mask] == tgt_pol.argmax(dim=1)[mask]).float().mean().item()
    )
    top1_conf = float(probs.max(dim=1).values.mean().item())  # decisiveness of the net
    # Effective number of moves the net spreads over (perplexity); higher = more diffuse.
    ent = float((-(probs * logp).sum(dim=1)).mean().item())
    perplexity = float(np.exp(ent))

    print(f"checkpoint: {args.checkpoint}  (n={n})")
    print("  [value]")
    print(f"    pred: mean={preds.mean():+.3f} std={pred_std:.3f} "
          f"min={preds.min():+.3f} max={preds.max():+.3f}")
    print(f"    target: mean={targs.mean():+.3f} std={targs.std():.3f}")
    print(f"    |pred|<0.1: {frac_near_zero*100:.1f}%   corr(pred,target): {v_corr:+.3f}")
    print(f"    -> {'VALUE HEAD LIKELY COLLAPSED' if v_collapsed else 'value head looks healthy'}")
    print("  [policy]  (vs Stockfish targets)")
    print(f"    top-1 agreement: {top1_agree*100:.1f}%")
    print(f"    cross-entropy:   {policy_ce:.3f}  (lower = closer to Stockfish)")
    print(f"    top-1 confidence: {top1_conf:.3f}   perplexity: {perplexity:.1f} moves")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from engine.inference import resolve_infer_backend
    from engine.player import EnginePlayer
    from train.evaluate import estimate_elo

    config = Config()
    checkpoint = args.checkpoint if args.checkpoint and os.path.exists(args.checkpoint) else None
    if not checkpoint:
        print("No checkpoint found; evaluating a randomly-initialized net.")
    infer = resolve_infer_backend(config, checkpoint)
    print(f"Inference: {infer}", flush=True)
    if args.workers <= 1:
        player = EnginePlayer(config=config, checkpoint=checkpoint, use_books=args.use_books)
        if args.use_books:
            print(
                f"Books: opening={'on' if player.opening_book else 'off'} "
                f"tablebase={'on' if player.tablebase else 'off'}",
                flush=True,
            )
    else:
        player = None
        print(f"Workers: {args.workers} (parallel games)", flush=True)
        if args.use_books:
            print(
                f"Books: opening={'on' if config.opening_book_path else 'off'} "
                f"tablebase={'on' if config.syzygy_path else 'off'}",
                flush=True,
            )

    def _progress(ev: dict) -> None:
        if ev.get("event") == "eval_game":
            elapsed = ev.get("elapsed_s")
            timing = f" {elapsed}s" if elapsed is not None else ""
            print(
                f"game {ev['game']}/{ev['games']} result={ev['result']} "
                f"running_score={ev['score']}{timing}",
                flush=True,
            )

    result = estimate_elo(
        player,
        config,
        games=args.games,
        sims=args.sims,
        skill_level=args.skill,
        movetime=args.movetime,
        progress=_progress,
        workers=args.workers,
        checkpoint=checkpoint,
        use_books=args.use_books,
    )
    print(result, flush=True)


def cmd_export_onnx(args: argparse.Namespace) -> None:
    from engine.inference import export_onnx, onnx_paths_for_checkpoint

    path = export_onnx(args.checkpoint, out_path=args.out, int8=args.int8)
    fp32, quant = onnx_paths_for_checkpoint(args.checkpoint, int8=True)
    print(f"Exported ONNX to {path}")
    if args.int8:
        print(f"  fp32 sidecar: {fp32}")
        print(f"  int8 model:   {quant}")
    print("Use with:")
    print(f"  set CHESSAI_INFER={'onnx-int8' if args.int8 else 'onnx'}")
    print(f"  set CHESSAI_ONNX={path}")


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

    sp = sub.add_parser("selfplay", help="Self-play refinement loop (arena-gated)")
    sp.add_argument("--iterations", type=int, default=40)
    sp.add_argument("--games-per-iter", type=int, default=48)
    sp.add_argument("--sims", type=int, default=400, help="MCTS sims per self-play move")
    sp.add_argument("--temperature-moves", type=int, default=25,
                    help="Plies with temperature=1.0 before switching to greedy play")
    sp.add_argument("--train-steps", type=int, default=200)
    sp.add_argument("--batch-size", type=int, default=256)
    sp.add_argument("--lr", type=float, default=1e-4, help="Learning rate (keep low to avoid forgetting)")
    sp.add_argument("--init", default="models/supervised.pt")
    sp.add_argument("--out", default="selfplay.pt")
    sp.add_argument("--sf-value-weight", type=float, default=0.0)
    sp.add_argument(
        "--buffer-capacity", type=int, default=300_000,
        help="Replay buffer size (raise for long runs, e.g. 300000)",
    )
    sp.add_argument(
        "--workers", type=int, default=1, help="Parallel self-play worker processes (e.g. 14)"
    )
    sp.add_argument(
        "--selfplay-device",
        default=None,
        help="Device for self-play workers (default: cpu when workers>1, else the training device)",
    )
    sp.add_argument(
        "--sup-fraction", type=float, default=0.5,
        help="Fraction of each training batch drawn from the labeled corpus (anti-forgetting)",
    )
    sp.add_argument("--no-resign", action="store_true",
                    help="Disable early resignation in self-play games")
    sp.add_argument("--resign-threshold", type=float, default=0.95,
                    help="Resign if value below -threshold for resign-streak plies")
    sp.add_argument("--resign-streak", type=int, default=3)
    sp.add_argument("--complete-fraction", type=float, default=0.08,
                    help="Fraction of self-play games that always play to completion")
    sp.add_argument("--arena-every", type=int, default=3, help="Run arena gating every N iters")
    sp.add_argument(
        "--arena-games", type=int, default=200,
        help="Games in the arena (played as mirrored pairs; use >=200 for reliable gating)",
    )
    sp.add_argument("--arena-sims", type=int, default=None,
                    help="MCTS sims for arena (default: same as --sims)")
    sp.add_argument("--arena-opening-pool", type=int, default=50,
                    help="Fixed pool of varied openings reused across arena pairs")
    sp.add_argument("--arena-book-fraction", type=float, default=0.7,
                    help="Fraction of arena openings sampled from the Polyglot book (rest random)")
    sp.add_argument("--arena-opening-min", type=int, default=6,
                    help="Min book-line plies when building opening pool")
    sp.add_argument("--arena-opening-max", type=int, default=24,
                    help="Max book-line plies when building opening pool")
    sp.add_argument(
        "--arena-temperature", type=float, default=0.0,
        help="Move-selection temperature for arena (0 = deterministic eval)",
    )
    sp.add_argument("--arena-temp-moves", type=int, default=0,
                    help="Plies the arena temperature applies for (0 = none)")
    sp.add_argument(
        "--gate-threshold", type=float, default=0.55,
        help="Candidate must score >= this vs champion to be promoted",
    )
    sp.add_argument(
        "--gate-min-games", type=int, default=200,
        help="Minimum arena games before a promotion decision is allowed",
    )
    sp.add_argument(
        "--no-gate-significance", action="store_true",
        help="Skip one-sided significance test (promote on score alone)",
    )
    sp.add_argument("--eval-every", type=int, default=6,
                    help="Absolute Elo check vs Stockfish every N iters (0=off)")
    sp.add_argument("--eval-games", type=int, default=20, help="Games per in-loop Elo eval")
    sp.add_argument("--eval-skill", type=int, default=5, help="Stockfish skill for in-loop eval")
    sp.add_argument("--eval-sims", type=int, default=None,
                    help="MCTS sims for in-loop eval (default: same as --sims)")
    sp.set_defaults(func=cmd_selfplay)

    pr = sub.add_parser("probe", help="Check the value head for collapse (pred vs target)")
    pr.add_argument("--checkpoint", default="models/selfplay.pt")
    pr.add_argument("--n", type=int, default=1024, help="Number of positions to sample")
    pr.set_defaults(func=cmd_probe)

    ex = sub.add_parser("export-onnx", help="Export checkpoint to ONNX (optional int8 quant)")
    ex.add_argument("--checkpoint", default="models/supervised_big.pt")
    ex.add_argument("--out", default=None, help="Output .onnx path (default: beside checkpoint)")
    ex.add_argument("--int8", action="store_true", help="Also write dynamic int8 quantized model")
    ex.set_defaults(func=cmd_export_onnx)

    e = sub.add_parser("evaluate", help="Estimate Elo vs Stockfish")
    e.add_argument("--checkpoint", default="models/best.pt")
    e.add_argument("--games", type=int, default=20)
    e.add_argument("--sims", type=int, default=80)
    e.add_argument("--skill", type=int, default=3)
    e.add_argument("--movetime", type=float, default=0.05)
    e.add_argument("--use-books", action="store_true",
                   help="Use opening book + tablebases (tests the full deployed engine)")
    e.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 4) // 2)),
        help="Parallel game workers (each loads its own net + Stockfish; default: ~half CPU cores, max 4)",
    )
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
