/* global $, Chess, Chessboard */
"use strict";

const PIECE_THEME = "/static/img/chesspieces/wikipedia/{piece}.png";

// --------------------------------------------------------------------------
// Tabs
// --------------------------------------------------------------------------
const editorInitialized = { done: false };

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    const tab = btn.dataset.tab;
    document.getElementById("tab-" + tab).classList.add("active");

    if (tab === "editor") {
      if (!editorInitialized.done) {
        initEditor();
        editorInitialized.done = true;
      } else if (editorBoard) {
        editorBoard.resize();
      }
    }
    if (tab === "game" && gameBoard) gameBoard.resize();
    if (tab === "training") refreshCheckpoints();
  });
});

async function api(path, method = "GET", body = null) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

// --------------------------------------------------------------------------
// Active checkpoint indicator
// --------------------------------------------------------------------------
async function refreshCheckpoints() {
  try {
    const data = await api("/api/checkpoints");
    const label =
      data.active ||
      (data.loaded ? "(untrained)" : "loading...");
    document.getElementById("active-checkpoint").textContent = label;
    const inferEl = document.getElementById("active-infer");
    if (inferEl) {
      const backend = data.infer_backend || "?";
      const path = data.infer_path ? ` (${data.infer_path.split(/[/\\]/).pop()})` : "";
      inferEl.textContent = backend + path;
    }
    const sel = document.getElementById("checkpoint-select");
    if (sel) {
      sel.innerHTML = "";
      data.checkpoints.forEach((c) => {
        const opt = document.createElement("option");
        opt.value = c;
        opt.textContent = c;
        sel.appendChild(opt);
      });
    }
  } catch (e) {
    /* server may still be starting */
  }
}

// --------------------------------------------------------------------------
// Game mode
// --------------------------------------------------------------------------
let gameBoard = null;
let game = new Chess();
let gameId = null;
let playerColor = "white";
let busy = false;

function onDragStart(source, piece) {
  if (busy || !gameId || game.game_over()) return false;
  const whiteToMove = game.turn() === "w";
  if ((playerColor === "white") !== whiteToMove) return false;
  if (playerColor === "white" && piece.startsWith("b")) return false;
  if (playerColor === "black" && piece.startsWith("w")) return false;
  return true;
}

function onDrop(source, target) {
  const move = game.move({ from: source, to: target, promotion: "q" });
  if (move === null) return "snapback";
  const uci = move.from + move.to + (move.promotion ? move.promotion : "");
  submitMove(uci);
}

function onSnapEnd() {
  gameBoard.position(game.fen());
}

async function submitMove(uci) {
  busy = true;
  setGameStatus("Engine is thinking...");
  try {
    const state = await api("/api/move", "POST", { game_id: gameId, uci });
    game.load(state.fen);
    gameBoard.position(state.fen);
    renderGameState(state);
  } catch (e) {
    setGameStatus("Error: " + e.message);
    // Reload authoritative state.
    if (gameId) {
      const state = await api("/api/state/" + gameId);
      game.load(state.fen);
      gameBoard.position(state.fen);
    }
  } finally {
    busy = false;
  }
}

function renderGameState(state) {
  const list = document.getElementById("move-list");
  const history = game.history();
  let html = "";
  for (let i = 0; i < history.length; i += 2) {
    const num = i / 2 + 1;
    const w = history[i] || "";
    const b = history[i + 1] || "";
    html += `<span>${num}. ${w} ${b}</span>`;
  }
  list.innerHTML = html;
  list.scrollTop = list.scrollHeight;

  if (state.game_over) {
    let msg = "Game over. Result: " + state.result;
    setGameStatus(msg);
  } else {
    const turn = state.turn === "white" ? "White" : "Black";
    const check = state.in_check ? " (check!)" : "";
    setGameStatus(`${turn} to move${check}`);
  }
}

function setGameStatus(text) {
  document.getElementById("game-status").textContent = text;
}

async function newGame() {
  playerColor = document.getElementById("game-color").value;
  const difficulty = document.getElementById("game-difficulty").value;
  busy = true;
  setGameStatus("Starting...");
  try {
    const state = await api("/api/new_game", "POST", {
      player_color: playerColor,
      difficulty,
    });
    gameId = state.game_id;
    game.load(state.fen);
    if (!gameBoard) initGameBoard();
    gameBoard.orientation(playerColor);
    gameBoard.position(state.fen);
    renderGameState(state);
  } catch (e) {
    setGameStatus("Error: " + e.message);
  } finally {
    busy = false;
  }
}

function initGameBoard() {
  gameBoard = Chessboard("game-board", {
    draggable: true,
    position: "start",
    pieceTheme: PIECE_THEME,
    onDragStart,
    onDrop,
    onSnapEnd,
  });
}

document.getElementById("new-game-btn").addEventListener("click", newGame);

// --------------------------------------------------------------------------
// Board editor
// --------------------------------------------------------------------------
let editorBoard = null;

function initEditor() {
  editorBoard = Chessboard("editor-board", {
    draggable: true,
    dropOffBoard: "trash",
    sparePieces: true,
    position: "start",
    pieceTheme: PIECE_THEME,
  });
}

function positionToFen(pos, turn) {
  const rows = [];
  for (let r = 8; r >= 1; r--) {
    let empty = 0;
    let row = "";
    for (let f = 0; f < 8; f++) {
      const sq = String.fromCharCode(97 + f) + r;
      const p = pos[sq];
      if (!p) {
        empty++;
      } else {
        if (empty > 0) {
          row += empty;
          empty = 0;
        }
        const color = p[0];
        const letter = p[1];
        row += color === "w" ? letter.toUpperCase() : letter.toLowerCase();
      }
    }
    if (empty > 0) row += empty;
    rows.push(row);
  }
  return rows.join("/") + " " + turn + " - - 0 1";
}

async function evaluatePosition() {
  const turn = document.getElementById("editor-turn").value;
  const fen = positionToFen(editorBoard.position(), turn);
  const summary = document.getElementById("eval-summary");
  const suggestions = document.getElementById("suggestions");
  summary.textContent = "Evaluating...";
  suggestions.innerHTML = "";
  try {
    const data = await api("/api/evaluate", "POST", { fen });
    if (data.game_over) {
      summary.innerHTML = `Position is terminal. Result: <b>${data.result}</b>`;
      return;
    }
    const whitePct = Math.round(
      (data.turn === "white" ? data.win_prob : 1 - data.win_prob) * 100
    );
    summary.innerHTML =
      `<div>Value (side to move): <b>${data.value}</b></div>` +
      `<div>Win probability: <b>${(data.win_prob * 100).toFixed(1)}%</b></div>` +
      `<div class="eval-bar" style="--white-pct:${whitePct}%"></div>` +
      `<div style="color:var(--muted)">White ${whitePct}% / Black ${100 - whitePct}%</div>`;
    data.suggestions.forEach((s) => {
      const li = document.createElement("li");
      li.innerHTML = `<b>${s.san}</b> &mdash; win ${(s.win_prob * 100).toFixed(
        1
      )}% <span style="color:var(--muted)">(${s.visits} visits)</span>`;
      suggestions.appendChild(li);
    });
    if (data.suggestions.length === 0) {
      suggestions.innerHTML = "<li>No legal moves.</li>";
    }
  } catch (e) {
    summary.textContent = "Error: " + e.message;
  }
}

document.getElementById("editor-start").addEventListener("click", () => editorBoard.start());
document.getElementById("editor-clear").addEventListener("click", () => editorBoard.clear());
document.getElementById("editor-flip").addEventListener("click", () => editorBoard.flip());
document.getElementById("evaluate-btn").addEventListener("click", evaluatePosition);

// --------------------------------------------------------------------------
// Training
// --------------------------------------------------------------------------
let eventSource = null;

function jobParams(job) {
  if (job === "label") {
    return {
      generate: document.getElementById("label-generate").value,
      depth: document.getElementById("label-depth").value,
      workers: document.getElementById("label-workers").value,
      multipv: document.getElementById("label-multipv").value,
    };
  }
  if (job === "supervised") {
    return {
      epochs: document.getElementById("sup-epochs").value,
      "batch-size": document.getElementById("sup-batch").value,
    };
  }
  if (job === "selfplay") {
    return {
      iterations: document.getElementById("sp-iters").value,
      "games-per-iter": document.getElementById("sp-games").value,
      sims: document.getElementById("sp-sims").value,
    };
  }
  return {};
}

async function startJob(job) {
  const log = document.getElementById("train-log");
  log.textContent = "";
  try {
    await api("/api/train/start", "POST", { mode: job, params: jobParams(job) });
    document.getElementById("train-status").textContent = `Running: ${job}`;
    openStream();
  } catch (e) {
    document.getElementById("train-status").textContent = "Error: " + e.message;
  }
}

function openStream() {
  if (eventSource) eventSource.close();
  const log = document.getElementById("train-log");
  eventSource = new EventSource("/api/train/stream");
  eventSource.onmessage = (ev) => {
    let data;
    try {
      data = JSON.parse(ev.data);
    } catch (_) {
      return;
    }
    if (data.event === "stream_end") {
      eventSource.close();
      document.getElementById("train-status").textContent = "Job finished.";
      refreshCheckpoints();
      return;
    }
    log.textContent += formatLogLine(data) + "\n";
    log.scrollTop = log.scrollHeight;
    updateTrainStatus(data);
  };
  eventSource.onerror = () => {
    /* stream closed */
  };
}

function formatLogLine(d) {
  const parts = [];
  Object.keys(d).forEach((k) => {
    if (k === "t") return;
    parts.push(`${k}=${d[k]}`);
  });
  return parts.join("  ");
}

function updateTrainStatus(d) {
  const status = document.getElementById("train-status");
  if (d.event === "epoch") {
    status.textContent = `Epoch ${d.epoch}: val policy ${d.val_policy_loss}, val value ${d.val_value_loss}`;
  } else if (d.event === "eval_done") {
    status.textContent = `Estimated Elo: ${d.estimated_elo} (score ${d.score})`;
  } else if (d.event === "selfplay_game") {
    status.textContent = `Self-play iter ${d.iter}: game ${d.game}/${d.games_per_iter}, buffer ${d.buffer}`;
  } else if (d.event === "generate") {
    if (d.status === "start") {
      status.textContent = `Generating ${d.n} positions (this can take a bit)...`;
    } else if (d.status === "progress") {
      status.textContent = `Generating positions ${d.n}/${d.target} — ${d.rate_pos_per_sec}/s`;
    } else if (d.status === "done") {
      status.textContent = `Generated ${d.n} positions in ${d.elapsed_s}s — starting Stockfish labeling...`;
    } else {
      status.textContent = `Loaded ${d.n} positions — starting labeling...`;
    }
  } else if (d.event === "error") {
    status.textContent = `Error (${d.stage}): ${d.message}`;
  } else if (d.event === "label") {
    status.textContent =
      `Labeling ${d.done}/${d.total} — ${d.rate_pos_per_sec}/s, ` +
      `elapsed ${d.elapsed}, ETA ${d.eta}`;
  } else if (d.event === "label_done") {
    status.textContent = `Labeling complete: ${d.done}/${d.total} in ${d.elapsed}`;
  } else if (d.event === "done") {
    status.textContent = `Done: ${d.checkpoint || d.mode}`;
  }
}

document.querySelectorAll("[data-job]").forEach((btn) => {
  btn.addEventListener("click", () => startJob(btn.dataset.job));
});

document.getElementById("stop-train").addEventListener("click", async () => {
  await api("/api/train/stop", "POST", {});
  document.getElementById("train-status").textContent = "Stopping...";
});

document.getElementById("refresh-checkpoints").addEventListener("click", refreshCheckpoints);
document.getElementById("activate-checkpoint").addEventListener("click", async () => {
  const name = document.getElementById("checkpoint-select").value;
  if (!name) return;
  try {
    const data = await api("/api/select_checkpoint", "POST", { name });
    document.getElementById("active-checkpoint").textContent = data.active;
  } catch (e) {
    alert("Error: " + e.message);
  }
});

// --------------------------------------------------------------------------
// Init
// --------------------------------------------------------------------------
initGameBoard();
refreshCheckpoints();
