"""FastAPI backend for the chess engine web app.

Serves three modes:
  - Board Editor: evaluate an arbitrary position, get suggested moves.
  - Game: play against the engine, choosing color and difficulty.
  - Training: launch/monitor labeling, supervised and self-play jobs (SSE).
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import subprocess
import sys
import threading
import uuid
from typing import Dict, Optional

import chess
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine.config import Config
from engine.model import build_model, load_checkpoint
from engine.player import EnginePlayer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
PROJECT_ROOT = os.path.dirname(BASE_DIR)

app = FastAPI(title="AlphaZero Chess Engine")

config = Config()
config.ensure_dirs()

DIFFICULTY_SIMS = {"easy": 30, "medium": 80, "hard": 200}


class EngineHolder:
    """Lazily loads a checkpoint and serves an :class:`EnginePlayer`."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._player: Optional[EnginePlayer] = None
        self.checkpoint_name: Optional[str] = None
        self.lock = threading.Lock()

    def _default_checkpoint(self) -> Optional[str]:
        env = os.environ.get("CHESSAI_CHECKPOINT")
        if env and os.path.exists(env):
            return env
        for name in ("best.pt", "selfplay.pt", "supervised.pt"):
            path = os.path.join(self.cfg.models_dir, name)
            if os.path.exists(path):
                return path
        return None

    def load(self, checkpoint: Optional[str] = None) -> None:
        with self.lock:
            checkpoint = checkpoint or self._default_checkpoint()
            if checkpoint and os.path.exists(checkpoint):
                model, _ = load_checkpoint(checkpoint, device=self.cfg.device)
                self.checkpoint_name = os.path.basename(checkpoint)
            else:
                model = build_model(self.cfg.model, device=self.cfg.device)
                self.checkpoint_name = "(untrained)"
            model.eval()
            self._player = EnginePlayer(model, self.cfg)

    @property
    def player(self) -> EnginePlayer:
        if self._player is None:
            self.load()
        assert self._player is not None
        return self._player


engine_holder = EngineHolder(config)
games: Dict[str, chess.Board] = {}
game_meta: Dict[str, dict] = {}


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class EvaluateRequest(BaseModel):
    fen: str
    sims: Optional[int] = None
    top_k: int = 5


class NewGameRequest(BaseModel):
    player_color: str = "white"  # 'white' or 'black'
    difficulty: str = "medium"


class MoveRequest(BaseModel):
    game_id: str
    uci: str


class SelectCheckpointRequest(BaseModel):
    name: str


class TrainRequest(BaseModel):
    mode: str  # 'label' | 'supervised' | 'selfplay'
    params: Dict[str, str] = {}


# --------------------------------------------------------------------------- #
# Static + index
# --------------------------------------------------------------------------- #
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# --------------------------------------------------------------------------- #
# Board editor / analysis
# --------------------------------------------------------------------------- #
@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest) -> dict:
    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN")
    if not board.is_valid():
        raise HTTPException(status_code=400, detail="Illegal position")
    result = engine_holder.player.evaluate_position(board, simulations=req.sims, top_k=req.top_k)
    result["fen"] = board.fen()
    result["turn"] = "white" if board.turn == chess.WHITE else "black"
    result["checkpoint"] = engine_holder.checkpoint_name
    return result


@app.get("/api/legal_moves")
def legal_moves(fen: str) -> dict:
    try:
        board = chess.Board(fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN")
    return {"moves": [m.uci() for m in board.legal_moves]}


# --------------------------------------------------------------------------- #
# Game play
# --------------------------------------------------------------------------- #
def _engine_move(board: chess.Board, difficulty: str) -> Optional[str]:
    if board.is_game_over(claim_draw=True):
        return None
    sims = DIFFICULTY_SIMS.get(difficulty, 80)
    move, _ = engine_holder.player.select_move(board, simulations=sims, temperature=0.0)
    board.push(move)
    return move.uci()


def _game_state(game_id: str) -> dict:
    board = games[game_id]
    meta = game_meta[game_id]
    outcome = board.outcome(claim_draw=True)
    return {
        "game_id": game_id,
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "player_color": meta["player_color"],
        "difficulty": meta["difficulty"],
        "game_over": board.is_game_over(claim_draw=True),
        "result": board.result(claim_draw=True) if outcome else "*",
        "last_move": meta.get("last_move"),
        "in_check": board.is_check(),
        "history": [m.uci() for m in board.move_stack],
    }


@app.post("/api/new_game")
def new_game(req: NewGameRequest) -> dict:
    game_id = uuid.uuid4().hex[:12]
    board = chess.Board()
    games[game_id] = board
    game_meta[game_id] = {
        "player_color": req.player_color,
        "difficulty": req.difficulty,
        "last_move": None,
    }
    # If the engine plays white, it moves first.
    if req.player_color == "black":
        engine_uci = _engine_move(board, req.difficulty)
        game_meta[game_id]["last_move"] = engine_uci
    state = _game_state(game_id)
    return state


@app.post("/api/move")
def move(req: MoveRequest) -> dict:
    if req.game_id not in games:
        raise HTTPException(status_code=404, detail="Unknown game")
    board = games[req.game_id]
    meta = game_meta[req.game_id]

    try:
        player_move = chess.Move.from_uci(req.uci)
    except ValueError:
        raise HTTPException(status_code=400, detail="Malformed move")
    if player_move not in board.legal_moves:
        raise HTTPException(status_code=400, detail="Illegal move")

    board.push(player_move)
    meta["last_move"] = player_move.uci()

    engine_uci = None
    if not board.is_game_over(claim_draw=True):
        engine_uci = _engine_move(board, meta["difficulty"])
        if engine_uci:
            meta["last_move"] = engine_uci

    state = _game_state(req.game_id)
    state["engine_move"] = engine_uci
    return state


@app.get("/api/state/{game_id}")
def state(game_id: str) -> dict:
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Unknown game")
    return _game_state(game_id)


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
@app.get("/api/checkpoints")
def checkpoints() -> dict:
    paths = sorted(glob.glob(os.path.join(config.models_dir, "*.pt")))
    return {
        "active": engine_holder.checkpoint_name,
        "checkpoints": [os.path.basename(p) for p in paths],
    }


@app.post("/api/select_checkpoint")
def select_checkpoint(req: SelectCheckpointRequest) -> dict:
    path = os.path.join(config.models_dir, req.name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    engine_holder.load(path)
    return {"active": engine_holder.checkpoint_name}


# --------------------------------------------------------------------------- #
# Training control (subprocess + SSE log tailing)
# --------------------------------------------------------------------------- #
class TrainingManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.proc: Optional[subprocess.Popen] = None
        self.mode: Optional[str] = None
        self.log_path: Optional[str] = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, mode: str, params: Dict[str, str]) -> None:
        if self.is_running():
            raise HTTPException(status_code=409, detail="A training job is already running")
        cmd = [sys.executable, "-m", "cli", mode]
        for k, v in params.items():
            cmd.append(f"--{k}")
            if v not in ("", None):
                cmd.append(str(v))
        self.mode = mode
        self.log_path = os.path.join(self.cfg.logs_dir, f"{mode}.jsonl")
        # ProgressLogger truncates this file when the job starts.
        env = dict(os.environ)
        self.proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            pid = self.proc.pid
            # Kill the whole tree: the label job spawns worker processes, each
            # with its own Stockfish child. A plain terminate() would orphan them.
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                )
            else:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass

    def status(self) -> dict:
        return {"running": self.is_running(), "mode": self.mode, "log": self.log_path}


training = TrainingManager(config)


@app.post("/api/train/start")
def train_start(req: TrainRequest) -> dict:
    if req.mode not in ("label", "supervised", "selfplay"):
        raise HTTPException(status_code=400, detail="Invalid mode")
    training.start(req.mode, req.params)
    return training.status()


@app.post("/api/train/stop")
def train_stop() -> dict:
    training.stop()
    return training.status()


@app.get("/api/train/status")
def train_status() -> dict:
    return training.status()


@app.get("/api/train/stream")
async def train_stream():
    log_path = training.log_path

    async def event_gen():
        pos = 0
        # Wait briefly for the log file to appear.
        for _ in range(50):
            if log_path and os.path.exists(log_path):
                break
            await asyncio.sleep(0.1)
        while True:
            if log_path and os.path.exists(log_path):
                with open(log_path, "r") as f:
                    f.seek(pos)
                    for line in f:
                        if line.strip():
                            yield f"data: {line.strip()}\n\n"
                    pos = f.tell()
            if not training.is_running():
                # Flush any remaining lines then stop.
                if log_path and os.path.exists(log_path):
                    with open(log_path, "r") as f:
                        f.seek(pos)
                        for line in f:
                            if line.strip():
                                yield f"data: {line.strip()}\n\n"
                yield 'data: {"event": "stream_end"}\n\n'
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream")
