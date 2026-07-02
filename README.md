# AlphaZero-Style Chess Engine

A from-scratch chess engine that learns the way AlphaZero / Lc0 do -- a
policy + value neural network guided by PUCT Monte Carlo Tree Search. To reach a
useful strength (~1500-2000 Elo) cheaply, it **bootstraps from Stockfish-labeled
positions** (supervised learning), then **refines via self-play**.

It ships with a web app offering three modes:

- **Play** - play the engine; choose to start as White or Black and pick a difficulty.
- **Board Editor** - place pieces anywhere, then get an evaluation and suggested moves.
- **Training** - launch and monitor labeling, supervised and self-play jobs live.

Everything auto-detects the compute device (CUDA / Apple MPS / CPU), so the same
code runs on a laptop CPU for iteration and scales to a cloud GPU for real training.

## Project layout

```
engine/     encoding, network model, MCTS, player, config
teacher/    Stockfish UCI wrapper (produces value + policy targets)
data/       position generation, Stockfish labeling, dataset + replay buffer
train/      supervised bootstrap, self-play loop, Elo evaluation, progress log
server/     FastAPI backend + static web frontend
cli.py      command-line entrypoint for the whole pipeline
```

## Setup

1. Install Python dependencies (a virtual environment is recommended):

   ```bash
   pip install -r requirements.txt
   ```

2. Install Stockfish (the teacher engine) and make it discoverable:

   - Download a binary from https://stockfishchess.org/download/ (or use a
     package manager: `brew install stockfish`, `apt install stockfish`,
     `choco install stockfish`).
   - Either put it on your `PATH` (so `stockfish` works) or point the engine at
     it explicitly:

     ```bash
     # macOS/Linux
     export STOCKFISH_PATH=/full/path/to/stockfish
     # Windows PowerShell
     $env:STOCKFISH_PATH = "C:\path\to\stockfish.exe"
     ```

   Stockfish is only needed for **training/evaluation**. You can run the web app
   and play against an (untrained) network without it.

## Quick start (end-to-end)

```bash
# 1. Label positions with Stockfish (generates 2000 positions, labels them).
python -m cli label --generate 2000 --depth 10

# 2. Supervised bootstrap on the labeled data -> models/supervised.pt
python -m cli supervised --epochs 8

# 3. (Optional) Self-play refinement -> models/selfplay.pt and models/best.pt
python -m cli selfplay --iterations 10 --games-per-iter 8 --sims 100

# 4. Estimate strength vs skill-limited Stockfish
python -m cli evaluate --checkpoint models/supervised.pt --games 20 --skill 3

# 5. Launch the web app
python -m cli serve --port 8000
# open http://127.0.0.1:8000
```

All of these steps can also be started and monitored from the **Training** tab
in the web UI.

## How training works

1. **Bootstrap (supervised).** `data/generate.py` produces a wide spread of
   positions; `teacher/stockfish.py` labels each with a value
   (`tanh(centipawns / 350)`, mates = +/-1) and a policy (softmax over the
   MultiPV moves). `train/supervised.py` trains the network to match them.
2. **Self-play refinement.** `train/selfplay.py` generates games with the
   network + MCTS (Dirichlet noise at the root, a temperature schedule for
   exploration). Training targets are the MCTS visit distribution (policy) and
   the game outcome (value). You can optionally blend the Stockfish evaluation
   into the value target with `--sf-value-weight` for extra stability.
3. **Evaluate + promote.** `train/evaluate.py` plays matches against Stockfish at
   a capped Skill Level and estimates Elo; the best checkpoint is saved to
   `models/best.pt`.

## Scaling to a cloud GPU

Nothing in the code is CPU-specific. On a GPU box, just install a CUDA build of
PyTorch and increase the knobs:

- Network size in `engine/config.py` (`ModelConfig.num_blocks`, `channels`).
- More labeled positions and higher Stockfish `--depth`.
- More self-play `--games-per-iter`, `--sims`, and `--iterations`.

The device is picked automatically, or force it with `CHESSAI_DEVICE=cuda`.

## Notes

- Promotions in the web UI auto-queen for simplicity.
- Elo numbers are approximate (anchored to rough Stockfish Skill-Level Elos); they
  are most reliable as a *relative* progress signal between checkpoints.
