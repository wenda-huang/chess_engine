"""Batched MCTS that runs entirely on the GPU.

Many independent games are searched in lockstep. Each simulation step advances *one* leaf
in every game's tree, so a step is a single batched network evaluation over all games.
The trees, the positions stored in them and the chess rules (``engine.gpuchess``) all
live in GPU tensors; Python only issues kernels.

One simulation step contains no host synchronisation and a fixed amount of work, so it is
recorded once as a HIP/CUDA graph and replayed ``sims`` times (falls back to eager launches
if graph capture is unavailable).

Value convention (same as ``engine.mcts``): a node's W/N is from the point of view of the
player to move *at that node*; a parent scores a child as ``-child.Q``.
"""
from __future__ import annotations

import os

# hipBLASLt cannot run under graph capture; plain hipBLAS can. Must be set before first GPU use.
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from engine import gpuchess as G
from engine.gpuchess import MAX_MOVES, Moves, State

DEPTH = 32                    # selection depth per simulation (deeper leaves are backed up as-is)
USE_GRAPHS = os.environ.get("CHESSAI_GRAPHS", "1") != "0"


@dataclass
class SearchResult:
    visits: torch.Tensor    # [B, A] root child visit counts (0 for invalid slots)
    code: torch.Tensor      # [B, A] packed root moves
    valid: torch.Tensor     # [B, A]
    pol: torch.Tensor       # [B, A] policy index of each root move
    root_q: torch.Tensor    # [B] mean value at the root (side-to-move view)
    root_value: torch.Tensor  # [B] raw network value at the root


class _Workspace:
    """Preallocated tree storage for up to ``B`` games and ``T`` nodes per game."""

    def __init__(self, B: int, T: int, device):
        A = MAX_MOVES
        self.B, self.T = B, T
        self.dummy_row = B * T   # scratch row for masked writes
        self.nodes = G.State.empty(B * T + 1, device)
        self.N = torch.zeros(B, T, device=device)
        self.W = torch.zeros(B, T, device=device)
        self.expanded = torch.zeros(B, T, dtype=torch.bool, device=device)
        self.term = torch.zeros(B, T, dtype=torch.bool, device=device)
        self.tval = torch.zeros(B, T, device=device)
        self.ccode = torch.zeros(B, T, A, dtype=torch.int32, device=device)
        self.cprior = torch.zeros(B, T, A, device=device)
        self.cchild = torch.full((B, T, A), -1, dtype=torch.int32, device=device)
        self.cvalid = torch.zeros(B, T, A, dtype=torch.bool, device=device)
        self.cpol = torch.zeros(B, T, A, dtype=torch.int32, device=device)
        self.counter = torch.ones(B, dtype=torch.long, device=device)
        self.path = torch.zeros(B, DEPTH + 2, dtype=torch.long, device=device)
        self.ar = torch.arange(B, device=device)
        self.graphs: Dict[tuple, torch.cuda.CUDAGraph] = {}


class _View:
    """First ``b`` games of a workspace (all tensors are leading-dimension slices)."""

    def __init__(self, ws: _Workspace, b: int):
        self.ws, self.b, self.T = ws, b, ws.T
        self.dummy_row = ws.dummy_row
        self.nodes = ws.nodes
        for name in ("N", "W", "expanded", "term", "tval", "ccode", "cprior", "cchild",
                     "cvalid", "cpol", "counter", "path", "ar"):
            setattr(self, name, getattr(ws, name)[:b])

    def reset(self) -> None:
        for t in (self.N, self.W, self.expanded, self.term, self.tval, self.cvalid):
            t.zero_()
        self.cchild.fill_(-1)
        self.counter.fill_(1)


_WS: Dict[Tuple[int, str], _Workspace] = {}


def _workspace(B: int, T: int, device) -> _View:
    key = (T, str(device))
    ws = _WS.get(key)
    if ws is None or ws.B < B:
        cap = ((B + 63) // 64) * 64
        _WS.clear()  # keep at most one big workspace alive (drops its graphs too)
        ws = _WS[key] = _Workspace(cap, T, device)
    view = _View(ws, B)
    view.reset()
    return view


@torch.no_grad()
def _evaluate(net, st: State, mv: Moves, amp: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Network priors over the legal moves ([B, A], sums to 1 over valid) and values [B]."""
    planes = G.to_planes(st)
    if amp:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, value = net(planes)
    else:
        logits, value = net(planes)
    logits = logits.float()
    lg = logits.gather(1, mv.pol)
    lg = lg.masked_fill(~mv.valid, float("-inf"))
    priors = torch.softmax(lg, dim=1)
    priors = torch.nan_to_num(priors, nan=0.0)  # rows with no legal moves
    return priors, value.float()


@torch.no_grad()
def _sim_step(net, ws: _View, c_puct: float, amp: bool) -> None:
    """One simulation for every game. No host syncs; all state changes are in-place."""
    b, T, ar = ws.b, ws.T, ws.ar
    N, W = ws.N, ws.W
    dev = N.device

    # ---- 1. selection: descend DEPTH levels (finished games just stop moving) -------------
    cur = torch.zeros(b, dtype=torch.long, device=dev)
    alive = torch.ones(b, dtype=torch.bool, device=dev)
    plen = torch.zeros(b, dtype=torch.long, device=dev)
    edge_parent = torch.zeros(b, dtype=torch.long, device=dev)
    edge_slot = torch.zeros(b, dtype=torch.long, device=dev)
    create = torch.zeros(b, dtype=torch.bool, device=dev)
    for d in range(DEPTH):
        ws.path[:, d] = cur
        plen = plen + alive.long()
        term_here = ws.term[ar, cur]
        ch = ws.cchild[ar, cur].long()                      # [b, A]
        has = ch >= 0
        chc = ch.clamp(min=0)
        cn = N.gather(1, chc) * has
        cw = W.gather(1, chc) * has
        q = torch.where(cn > 0, -cw / cn.clamp(min=1.0), torch.zeros_like(cn))
        n_cur = N[ar, cur]
        u = c_puct * ws.cprior[ar, cur] * torch.sqrt(n_cur + 1.0)[:, None] / (1.0 + cn)
        score = torch.where(ws.cvalid[ar, cur], q + u, torch.full_like(q, float("-inf")))
        a = score.argmax(1)
        nxt = ch.gather(1, a[:, None]).squeeze(1)
        descend = alive & ~term_here
        missing = descend & (nxt < 0)
        create = create | missing
        edge_parent = torch.where(missing, cur, edge_parent)
        edge_slot = torch.where(missing, a, edge_slot)
        go = descend & (nxt >= 0)
        cur = torch.where(go, nxt, cur)
        alive = go
    ws.path[:, DEPTH] = cur          # games still descending: ``cur`` is the (existing) leaf
    plen = plen + alive.long()

    # ---- 2. expand newly reached leaves --------------------------------------------------------
    new_id = ws.counter.clone()
    ws.counter += create.long()
    pst = ws.nodes.index(ar * T + edge_parent)
    code = ws.ccode[ar, edge_parent, edge_slot]
    new_st = G.make_moves(pst, code)
    nmv = G.legal_moves(new_st, exact=False)
    is_term, tv = G.terminal_info(new_st, nmv)
    npriors, nvalue = _evaluate(net, new_st, nmv, amp)

    wrow = torch.where(create, ar * T + new_id, torch.full_like(ar, ws.dummy_row))
    ws.nodes.sq[wrow] = new_st.sq
    ws.nodes.stm[wrow] = new_st.stm
    ws.nodes.castle[wrow] = new_st.castle
    ws.nodes.ep[wrow] = new_st.ep
    ws.nodes.half[wrow] = new_st.half
    nid = torch.where(create, new_id, torch.zeros_like(new_id))   # non-created rows rewrite node 0 unchanged
    cm = create
    ws.expanded[ar, nid] = torch.where(cm, torch.ones_like(cm), ws.expanded[ar, nid])
    ws.term[ar, nid] = torch.where(cm, is_term, ws.term[ar, nid])
    ws.tval[ar, nid] = torch.where(cm, tv, ws.tval[ar, nid])
    ws.ccode[ar, nid] = torch.where(cm[:, None], nmv.code, ws.ccode[ar, nid])
    ws.cprior[ar, nid] = torch.where(cm[:, None], npriors, ws.cprior[ar, nid])
    ws.cvalid[ar, nid] = torch.where(cm[:, None], nmv.valid & ~is_term[:, None], ws.cvalid[ar, nid])
    ws.cpol[ar, nid] = torch.where(cm[:, None], nmv.pol.to(torch.int32), ws.cpol[ar, nid])
    link = ws.cchild[ar, edge_parent, edge_slot]
    ws.cchild[ar, edge_parent, edge_slot] = torch.where(cm, new_id.to(torch.int32), link)

    # ---- 3. back up ------------------------------------------------------------------------------
    leaf_val = torch.where(is_term, tv, nvalue)
    q_cur = W[ar, cur] / N[ar, cur].clamp(min=1.0)
    revisit_val = torch.where(ws.term[ar, cur], ws.tval[ar, cur], q_cur)   # existing leaf (terminal / depth cap)
    v = torch.where(create, leaf_val, revisit_val)
    wi = plen.clamp(max=DEPTH + 1)
    ws.path[ar, wi] = torch.where(create, new_id, ws.path[ar, wi])
    total = plen + create.long()             # nodes on the path, leaf last
    cols = torch.arange(DEPTH + 2, device=dev)[None, :]
    m = cols < total[:, None]
    sign = torch.where(((total[:, None] - 1 - cols) % 2) == 0, 1.0, -1.0)
    flat = (ar[:, None] * T + ws.path).reshape(-1)
    N.view(-1).scatter_add_(0, flat, m.float().reshape(-1))
    W.view(-1).scatter_add_(0, flat, torch.where(m, v[:, None] * sign, torch.zeros_like(sign)).reshape(-1))


def _get_graph(net, ws: _View, c_puct: float, amp: bool):
    """Record (once per net / batch size / settings) one simulation step as a graph."""
    key = (id(net), ws.b, c_puct, amp)
    g = ws.ws.graphs.get(key)
    if g is not None:
        return g
    dev = ws.N.device
    # Warm up every kernel/library the step touches on a scratch batch (no tree state is changed).
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        dummy = G.start_state(ws.b, dev)
        for _ in range(2):
            mv = G.legal_moves(dummy, exact=False)
            G.terminal_info(dummy, mv)
            _evaluate(net, G.make_moves(dummy, mv.code[:, 0]), mv, amp)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _sim_step(net, ws, c_puct, amp)
    torch.cuda.synchronize()
    ws.ws.graphs[key] = g
    return g


def _disable_graphs(err: Exception) -> None:
    global USE_GRAPHS
    USE_GRAPHS = False
    print(f"[gpumcts] graph capture failed ({type(err).__name__}: {str(err)[:160]}); using eager launches",
          flush=True)


@torch.no_grad()
def search(
    net,
    st: State,
    sims: int,
    c_puct: float = 1.5,
    add_noise: bool = False,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    amp: bool = False,
) -> SearchResult:
    """Run ``sims`` simulations from each root in ``st`` (all roots must be non-terminal)."""
    dev = st.device
    B = len(st)
    T = sims + 1
    A = MAX_MOVES
    ws = _workspace(B, T, dev)
    ar = ws.ar

    # ---- root ---------------------------------------------------------------------------
    mv = G.legal_moves(st, exact=True)
    priors, value = _evaluate(net, st, mv, amp)
    root_value = value
    if add_noise:
        gam = torch.distributions.Gamma(
            torch.full((B, A), dirichlet_alpha, device=dev), torch.ones((B, A), device=dev)
        ).sample()
        gam = gam * mv.valid
        noise = gam / gam.sum(1, keepdim=True).clamp(min=1e-12)
        priors = (1 - dirichlet_epsilon) * priors + dirichlet_epsilon * noise
    rows = ar * T
    ws.nodes.sq[rows] = st.sq
    ws.nodes.stm[rows] = st.stm
    ws.nodes.castle[rows] = st.castle
    ws.nodes.ep[rows] = st.ep
    ws.nodes.half[rows] = st.half
    ws.expanded[:, 0] = True
    ws.ccode[:, 0] = mv.code
    ws.cprior[:, 0] = priors
    ws.cvalid[:, 0] = mv.valid
    ws.cpol[:, 0] = mv.pol.to(torch.int32)

    graph = None
    if USE_GRAPHS and dev.type == "cuda":
        try:
            graph = _get_graph(net, ws, c_puct, amp)
        except Exception as e:  # capture unsupported here: fall back to eager launches
            _disable_graphs(e)
    if graph is not None:
        for _ in range(sims):
            graph.replay()
    else:
        for _ in range(sims):
            _sim_step(net, ws, c_puct, amp)

    N, W = ws.N, ws.W
    ch0 = ws.cchild[:, 0].long()
    has0 = ch0 >= 0
    visits = N.gather(1, ch0.clamp(min=0)) * has0
    root_q = W[:, 0] / N[:, 0].clamp(min=1.0)
    return SearchResult(visits, ws.ccode[:, 0].clone(), ws.cvalid[:, 0].clone(),
                        ws.cpol[:, 0].long().clone(), root_q, root_value)
