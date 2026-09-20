"""GPU-resident self-play, arena matches and replay buffer.

Everything from move generation to search to the training data lives on the GPU. The
host only launches kernels and, once per finished batch of games, reads back a few scalars
for logging.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import chess
import torch

from engine import gpuchess as G
from engine import gpumcts as M
from engine.encoding import POLICY_SIZE
from engine.gpuchess import MAX_MOVES, State


# --------------------------------------------------------------------------- #
# Replay buffer (sparse policies, all on the GPU)
# --------------------------------------------------------------------------- #
class GpuReplayBuffer:
    """Ring buffer of (planes uint8, sparse visit policy, value target) on the GPU."""

    def __init__(self, capacity: int, device):
        self.capacity = capacity
        self.device = device
        self.planes = torch.zeros(capacity, 18, 8, 8, dtype=torch.uint8, device=device)
        self.pol = torch.zeros(capacity, MAX_MOVES, dtype=torch.int16, device=device)
        self.prob = torch.zeros(capacity, MAX_MOVES, dtype=torch.float16, device=device)
        self.z = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, planes: torch.Tensor, pol: torch.Tensor, prob: torch.Tensor, z: torch.Tensor) -> None:
        n = planes.shape[0]
        if n == 0:
            return
        if n > self.capacity:
            planes, pol, prob, z = planes[-self.capacity:], pol[-self.capacity:], prob[-self.capacity:], z[-self.capacity:]
            n = self.capacity
        idx = (self.pos + torch.arange(n, device=self.device)) % self.capacity
        self.planes[idx] = planes
        self.pol[idx] = pol
        self.prob[idx] = prob
        self.z[idx] = z
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample_batch(self, n: int):
        idx = torch.randint(0, self.size, (min(n, self.size),), device=self.device)
        planes = self.planes[idx].float()
        policy = torch.zeros(len(idx), POLICY_SIZE, device=self.device)
        policy.scatter_add_(1, self.pol[idx].long(), self.prob[idx].float())
        return planes, policy, self.z[idx]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _repetition(hist: torch.Tensor, ply: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
    """Threefold repetition of the current position within each slot's history."""
    idx = torch.arange(hist.shape[1], device=hist.device)[None, :]
    same = (hist == cur[:, None]) & (idx <= ply[:, None])
    return same.sum(1) >= 3


def _game_over(st: State, ply, hist, max_plies) -> Tuple[torch.Tensor, torch.Tensor]:
    """(over, white_result) for positions where the game ended by the rules."""
    mv = G.legal_moves(st)
    term, val = G.terminal_info(st, mv)
    mate = term & (val < 0)
    rep = _repetition(hist, ply, G.hash_state(st))
    over = term | rep | (ply >= max_plies)
    white_result = torch.where(mate, torch.where(st.stm, -1.0, 1.0), torch.zeros_like(val))
    return over, white_result


def _pick(res: M.SearchResult, sample: torch.Tensor) -> torch.Tensor:
    """Choose a root move: sample proportionally to visits where ``sample`` else argmax."""
    total = res.visits.sum(1, keepdim=True)
    probs = torch.where(total > 0, res.visits / total.clamp(min=1.0), torch.ones_like(res.visits))  # dead rows: uniform
    greedy = res.visits.argmax(1)
    drawn = torch.multinomial(probs, 1).squeeze(1)
    return torch.where(sample, drawn, greedy)


# --------------------------------------------------------------------------- #
# Self-play
# --------------------------------------------------------------------------- #
@torch.no_grad()
def gpu_selfplay(
    net,
    buffer: GpuReplayBuffer,
    n_games: int,
    slots: int = 256,
    sims: int = 400,
    temperature_moves: int = 18,
    max_moves: int = 200,
    resign: bool = True,
    resign_threshold: float = 0.95,
    resign_streak: int = 3,
    complete_fraction: float = 0.08,
    c_puct: float = 1.5,
    amp: bool = False,
    on_game: Optional[Callable[[dict], None]] = None,
    on_tick: Optional[Callable[[], None]] = None,
) -> Dict[str, float]:
    """Play ``n_games`` self-play games and append every position to ``buffer``."""
    dev = buffer.device
    S = max(1, min(slots, n_games))
    A = MAX_MOVES
    ar = torch.arange(S, device=dev)
    net.eval()

    st = G.start_state(S, dev)
    start = G.start_state(S, dev)
    ply = torch.zeros(S, dtype=torch.long, device=dev)
    alive = torch.ones(S, dtype=torch.bool, device=dev)
    started = S
    hist = torch.zeros(S, max_moves + 2, dtype=torch.long, device=dev)
    hist[:, 0] = G.hash_state(st)
    bad = torch.zeros(S, 2, dtype=torch.long, device=dev)
    forced = torch.full((S,), float("nan"), device=dev)          # resignation result, NaN = none
    complete = (torch.rand(S, device=dev) < complete_fraction) | (not resign)

    hp = torch.zeros(S, max_moves + 1, 18, 8, 8, dtype=torch.uint8, device=dev)
    hpol = torch.zeros(S, max_moves + 1, A, dtype=torch.int16, device=dev)
    hprob = torch.zeros(S, max_moves + 1, A, dtype=torch.float16, device=dev)
    hstm = torch.zeros(S, max_moves + 1, dtype=torch.bool, device=dev)
    tidx = torch.arange(max_moves + 1, device=dev)[None, :]

    finished = 0
    stats = {"white": 0, "black": 0, "draw": 0, "samples": 0}

    n_iter = 0
    while bool(alive.any()):
        n_iter += 1
        if on_tick is not None and n_iter % 10 == 0:
            on_tick()
        # ---- finish games that ended (by rules, ply cap or resignation) --------------------
        over_rules, wres = _game_over(st, ply, hist, max_moves)
        resigned = ~torch.isnan(forced)
        over = alive & (over_rules | resigned)
        wres = torch.where(resigned, forced, wres)
        if bool(over.any()):
            mask = (tidx < ply[:, None]) & over[:, None]
            rs, ts = mask.nonzero(as_tuple=True)
            z = wres[rs] * torch.where(hstm[rs, ts], 1.0, -1.0)
            buffer.add(hp[rs, ts], hpol[rs, ts], hprob[rs, ts], z)
            stats["samples"] += int(rs.numel())
            done_slots = over.nonzero().squeeze(1)
            results = wres[done_slots].tolist()
            lengths = ply[done_slots].tolist()
            for r, n in zip(results, lengths):
                finished += 1
                stats["white" if r > 0 else "black" if r < 0 else "draw"] += 1
                if on_game is not None:
                    on_game({"game": finished, "samples": int(n), "result": float(r), "buffer": len(buffer)})
            # refill with new games while the quota lasts
            n_new = min(int(over.sum()), n_games - started)
            refill = torch.zeros(S, dtype=torch.bool, device=dev)
            if n_new > 0:
                refill[done_slots[:n_new]] = True
            started += n_new
            alive = alive & ~(over & ~refill)
            st.assign(refill, start)
            ply = torch.where(refill, torch.zeros_like(ply), ply)
            hist[refill, 0] = G.hash_state(start)[refill]
            bad[refill] = 0
            complete = torch.where(refill, (torch.rand(S, device=dev) < complete_fraction) | (not resign), complete)
            forced = torch.where(over, torch.full_like(forced, float("nan")), forced)
            st.assign(~alive, start)   # park finished slots on a harmless position
            if not bool(alive.any()):
                break

        # ---- search + move --------------------------------------------------------------
        res = M.search(net, st, sims, c_puct=c_puct, add_noise=True, amp=amp)
        sel = _pick(res, ply < temperature_moves)
        code = res.code.gather(1, sel[:, None]).squeeze(1)

        # record training samples for every live slot (position, visit policy, side to move)
        write_t = ply.clamp(max=max_moves)
        planes = G.to_planes(st).to(torch.uint8)
        total = res.visits.sum(1, keepdim=True).clamp(min=1.0)
        hp[ar, write_t] = planes
        hpol[ar, write_t] = res.pol.to(torch.int16)
        hprob[ar, write_t] = (res.visits / total).to(torch.float16)
        hstm[ar, write_t] = st.stm

        # resignation (checked after recording, like the CPU implementation)
        resign_now = torch.zeros(S, dtype=torch.bool, device=dev)
        if resign:
            color = (~st.stm).long()
            hit = alive & ~complete & (res.root_q < -resign_threshold)
            cur_bad = bad[ar, color]
            cur_bad = torch.where(hit, cur_bad + 1, torch.zeros_like(cur_bad))
            bad[ar, color] = cur_bad
            resign_now = hit & (cur_bad >= resign_streak)
            forced = torch.where(resign_now, torch.where(st.stm, -1.0, 1.0), forced)

        advance = alive & ~resign_now
        new_st = G.make_moves(st, code)
        st.assign(advance, new_st)
        ply = ply + (advance | resign_now).long()   # a resigning slot keeps its final sample
        nh = G.hash_state(st)
        hist[ar, ply.clamp(max=max_moves + 1)] = torch.where(advance, nh, hist[ar, ply.clamp(max=max_moves + 1)])

    return {k: float(v) for k, v in stats.items()}


# --------------------------------------------------------------------------- #
# Arena
# --------------------------------------------------------------------------- #
@torch.no_grad()
def gpu_arena(
    cand,
    champ,
    openings: List[List[chess.Move]],
    sims: int = 400,
    max_plies: int = 300,
    temperature_moves: int = 0,
    c_puct: float = 1.5,
    amp: bool = False,
    on_game: Optional[Callable[[dict], None]] = None,
    on_tick: Optional[Callable[[], None]] = None,
) -> Tuple[float, int, int, int]:
    """Candidate vs champion, every opening played twice with colours swapped.

    Returns (candidate_score, wins, draws, losses) over ``2 * len(openings)`` games.
    All games run concurrently on the GPU; both networks search every ply (two batched
    searches) and each game keeps the move of whichever network is to move.
    """
    dev = next(cand.parameters()).device
    cand.eval()
    champ.eval()
    P = len(openings)
    S = 2 * P
    boards = []
    for op in openings:
        b = chess.Board()
        for m in op:
            if m in b.legal_moves:
                b.push(m)
        boards.append(b)
    st = G.state_from_boards([b for b in boards for _ in range(2)], dev)  # slots 2i, 2i+1 share opening i
    cand_white = (torch.arange(S, device=dev) % 2 == 0)                   # even slot: candidate has White
    ar = torch.arange(S, device=dev)
    ply = torch.zeros(S, dtype=torch.long, device=dev)
    hist = torch.zeros(S, max_plies + 2, dtype=torch.long, device=dev)
    hist[:, 0] = G.hash_state(st)
    done = torch.zeros(S, dtype=torch.bool, device=dev)
    result = torch.zeros(S, device=dev)            # white-perspective result

    n_iter = 0
    while True:
        n_iter += 1
        if on_tick is not None and n_iter % 5 == 0:
            on_tick()
        over_rules, wres = _game_over(st, ply, hist, max_plies)
        newly = ~done & over_rules
        result = torch.where(newly, wres, result)
        done = done | newly
        if bool(done.all()):
            break
        cand_moves = (st.stm == cand_white)
        code = torch.zeros(S, dtype=torch.int32, device=dev)
        for net_, who in ((cand, cand_moves & ~done), (champ, ~cand_moves & ~done)):
            idx = who.nonzero().squeeze(1)
            n = int(idx.numel())
            if n == 0:
                continue
            padded = ((n + 63) // 64) * 64     # few distinct batch shapes -> kernels stay warm
            pidx = torch.cat([idx, idx[:1].expand(padded - n)]) if padded > n else idx
            res = M.search(net_, st.index(pidx), sims, c_puct=c_puct, add_noise=False, amp=amp)
            sel = _pick(res, ply[pidx] < temperature_moves)
            code[idx] = res.code.gather(1, sel[:, None]).squeeze(1)[:n]
        adv = ~done
        new_st = G.make_moves(st, code)
        st.assign(adv, new_st)
        ply = ply + adv.long()
        nh = G.hash_state(st)
        idx = ply.clamp(max=max_plies + 1)
        hist[ar, idx] = torch.where(adv, nh, hist[ar, idx])

    r = result.view(P, 2)                          # [pair, (cand white, cand black)]
    cand_scores = torch.stack([(r[:, 0] + 1) / 2, (1 - r[:, 1]) / 2], dim=1).reshape(-1)
    wins = int((cand_scores == 1.0).sum())
    draws = int((cand_scores == 0.5).sum())
    losses = int((cand_scores == 0.0).sum())
    if on_game is not None:
        for i in range(P):
            on_game({"white_result_a": float(r[i, 0]), "white_result_b": float(r[i, 1]),
                     "cand_pair_score": float(cand_scores[2 * i] + cand_scores[2 * i + 1])})
    return float(cand_scores.mean()), wins, draws, losses
