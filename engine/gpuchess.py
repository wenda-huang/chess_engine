"""Batched chess rules on the GPU (torch tensors only).

A ``State`` holds B positions as tensors. Move generation, move making, terminal
detection and plane encoding are fully vectorised, so a whole batch of games can
be advanced without any per-position Python work.

Conventions (identical to python-chess / engine.encoding):
  * square = rank * 8 + file (a1 = 0, h1 = 7, a8 = 56)
  * piece codes: 0 empty, 1..6 = white P N B R Q K, 7..12 = black P N B R Q K
  * castle rights order: [white kingside, white queenside, black kingside, black queenside]
  * ``ep`` is the square *behind* a pawn that just double-pushed (-1 if none), set after
    every double push, like ``chess.Board.ep_square``
  * a move is packed into an int32: ``from | to << 6 | promo << 12`` where promo is
    0 (none / queen), 1 knight, 2 bishop, 3 rook. This matches the policy vocabulary in
    engine.encoding (queen promotions fold into the plain from/to move).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import chess

from engine.encoding import IDX_TO_MOVE, MOVE_TO_IDX, POLICY_SIZE

MAX_MOVES = 256        # legal move slots per position (the maximum possible is 218)
_PSEUDO_FAST = 160     # pseudo-legal candidates checked per position in the fast pass
_PSEUDO_FULL = 320     # fallback for the rare positions with more pseudo-legal moves
_NSLOT = 78            # candidate slots per from-square (see _tables)
PROMO_PIECE = [5, 2, 3, 4]  # promo code -> white piece type (Q, N, B, R)


# --------------------------------------------------------------------------- #
# Static tables
# --------------------------------------------------------------------------- #
def _build_tables() -> Dict[str, np.ndarray]:
    ray = np.full((64, 8, 7), 64, dtype=np.int64)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for s in range(64):
        r, f = divmod(s, 8)
        for d, (dr, df) in enumerate(dirs):
            for k in range(1, 8):
                rr, ff = r + dr * k, f + df * k
                if 0 <= rr < 8 and 0 <= ff < 8:
                    ray[s, d, k - 1] = rr * 8 + ff

    kn = np.full((64, 8), 64, dtype=np.int64)
    kg = np.full((64, 8), 64, dtype=np.int64)
    kn_off = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
    kg_off = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for s in range(64):
        r, f = divmod(s, 8)
        for i, (dr, df) in enumerate(kn_off):
            if 0 <= r + dr < 8 and 0 <= f + df < 8:
                kn[s, i] = (r + dr) * 8 + f + df
        for i, (dr, df) in enumerate(kg_off):
            if 0 <= r + dr < 8 and 0 <= f + df < 8:
                kg[s, i] = (r + dr) * 8 + f + df

    # pawn targets per colour: [single, double, capture-left, capture-right]
    pawn = np.full((2, 64, 4), 64, dtype=np.int64)
    pawn_att = np.full((2, 64, 2), 64, dtype=np.int64)  # squares from which a pawn attacks t
    for c, (dr, start) in enumerate(((1, 1), (-1, 6))):
        for s in range(64):
            r, f = divmod(s, 8)
            if r in (0, 7):
                continue
            if 0 <= r + dr < 8:
                pawn[c, s, 0] = (r + dr) * 8 + f
            if r == start:
                pawn[c, s, 1] = (r + 2 * dr) * 8 + f
            for j, df in enumerate((-1, 1)):
                if 0 <= r + dr < 8 and 0 <= f + df < 8:
                    pawn[c, s, 2 + j] = (r + dr) * 8 + f + df
        cnt = [0] * 64
        for s in range(64):
            for j in (2, 3):
                t = pawn[c, s, j]
                if t < 64:
                    pawn_att[c, t, cnt[t]] = s
                    cnt[t] += 1

    castle_to = np.full((64, 2), 64, dtype=np.int64)
    castle_to[4] = (6, 2)
    castle_to[60] = (62, 58)

    clear = np.zeros((64, 4), dtype=bool)
    for sq_, idxs in ((4, (0, 1)), (7, (0,)), (0, (1,)), (60, (2, 3)), (63, (2,)), (56, (3,))):
        for i in idxs:
            clear[sq_, i] = True

    # (from, to, promo) -> policy index. Plain vocabulary entries cover queen promotions.
    lookup = np.full((4, 4096), -1, dtype=np.int64)
    for (frm, to, promo), idx in MOVE_TO_IDX.items():
        p = 0 if promo is None else {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3}[promo]
        lookup[p, to * 64 + frm] = idx  # index by (code & 4095) = from | to << 6

    return {
        "ray": ray, "kn": kn, "kg": kg, "pawn": pawn, "pawn_att": pawn_att,
        "castle_to": castle_to, "clear": clear, "lookup": lookup,
    }


_TABLES: Dict[str, Dict[str, torch.Tensor]] = {}
_NP_TABLES: Optional[Dict[str, np.ndarray]] = None


def tables(device) -> Dict[str, torch.Tensor]:
    global _NP_TABLES
    key = str(device)
    if key not in _TABLES:
        if _NP_TABLES is None:
            _NP_TABLES = _build_tables()
        t = {k: torch.from_numpy(v).to(device) for k, v in _NP_TABLES.items()}
        # slot -> target square for each colour, [2, 64, _NSLOT]
        ray = t["ray"].reshape(64, 56)
        common = torch.cat([ray, t["kn"], t["kg"]], dim=1)  # 72 slots
        slot_to = torch.stack(
            [torch.cat([common, t["pawn"][c], t["castle_to"]], dim=1) for c in range(2)]
        )
        t["slot_to"] = slot_to
        t["promo_piece"] = torch.tensor(PROMO_PIECE, device=device, dtype=torch.int8)
        rng = np.random.RandomState(12345)
        t["zob_sq"] = torch.from_numpy(
            rng.randint(-2 ** 62, 2 ** 62, size=(64, 13), dtype=np.int64)).to(device)
        t["zob_castle"] = torch.from_numpy(
            rng.randint(-2 ** 62, 2 ** 62, size=(4,), dtype=np.int64)).to(device)
        t["zob_ep"] = torch.from_numpy(
            rng.randint(-2 ** 62, 2 ** 62, size=(65,), dtype=np.int64)).to(device)
        t["zob_stm"] = torch.from_numpy(
            rng.randint(-2 ** 62, 2 ** 62, size=(1,), dtype=np.int64)).to(device)
        t["arange64"] = torch.arange(64, device=device)
        _TABLES[key] = t
    return _TABLES[key]


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
@dataclass
class State:
    sq: torch.Tensor      # [B, 64] int8
    stm: torch.Tensor     # [B] bool   (True = white to move)
    castle: torch.Tensor  # [B, 4] bool
    ep: torch.Tensor      # [B] int64  (-1 = none)
    half: torch.Tensor    # [B] int64  halfmove clock

    def __len__(self) -> int:
        return self.sq.shape[0]

    @property
    def device(self):
        return self.sq.device

    def index(self, idx: torch.Tensor) -> "State":
        return State(self.sq[idx], self.stm[idx], self.castle[idx], self.ep[idx], self.half[idx])

    def clone(self) -> "State":
        return State(self.sq.clone(), self.stm.clone(), self.castle.clone(),
                     self.ep.clone(), self.half.clone())

    def assign(self, mask: torch.Tensor, other: "State") -> None:
        """Overwrite the rows where ``mask`` is True with the same rows of ``other``."""
        self.sq = torch.where(mask[:, None], other.sq, self.sq)
        self.stm = torch.where(mask, other.stm, self.stm)
        self.castle = torch.where(mask[:, None], other.castle, self.castle)
        self.ep = torch.where(mask, other.ep, self.ep)
        self.half = torch.where(mask, other.half, self.half)

    @staticmethod
    def empty(n: int, device) -> "State":
        return State(
            torch.zeros(n, 64, dtype=torch.int8, device=device),
            torch.ones(n, dtype=torch.bool, device=device),
            torch.zeros(n, 4, dtype=torch.bool, device=device),
            torch.full((n,), -1, dtype=torch.long, device=device),
            torch.zeros(n, dtype=torch.long, device=device),
        )


def state_from_boards(boards: List[chess.Board], device) -> State:
    n = len(boards)
    sq = np.zeros((n, 64), dtype=np.int8)
    stm = np.zeros(n, dtype=bool)
    castle = np.zeros((n, 4), dtype=bool)
    ep = np.full(n, -1, dtype=np.int64)
    half = np.zeros(n, dtype=np.int64)
    for i, b in enumerate(boards):
        for s, p in b.piece_map().items():
            sq[i, s] = p.piece_type + (0 if p.color == chess.WHITE else 6)
        stm[i] = b.turn == chess.WHITE
        castle[i] = [b.has_kingside_castling_rights(chess.WHITE),
                     b.has_queenside_castling_rights(chess.WHITE),
                     b.has_kingside_castling_rights(chess.BLACK),
                     b.has_queenside_castling_rights(chess.BLACK)]
        ep[i] = -1 if b.ep_square is None else b.ep_square
        half[i] = b.halfmove_clock
    return State(torch.from_numpy(sq).to(device), torch.from_numpy(stm).to(device),
                 torch.from_numpy(castle).to(device), torch.from_numpy(ep).to(device),
                 torch.from_numpy(half).to(device))


def start_state(n: int, device) -> State:
    return state_from_boards([chess.Board()], device).index(torch.zeros(n, dtype=torch.long, device=device))


def board_from_state(st: State, i: int) -> chess.Board:
    """Rebuild a python-chess board (for tests / debugging)."""
    b = chess.Board(None)
    sq = st.sq[i].cpu().numpy()
    for s in range(64):
        p = int(sq[s])
        if p:
            b.set_piece_at(s, chess.Piece((p - 1) % 6 + 1, chess.WHITE if p <= 6 else chess.BLACK))
    b.turn = bool(st.stm[i])
    rights = 0
    c = st.castle[i].cpu().numpy()
    if c[0]:
        rights |= chess.BB_H1
    if c[1]:
        rights |= chess.BB_A1
    if c[2]:
        rights |= chess.BB_H8
    if c[3]:
        rights |= chess.BB_A8
    b.castling_rights = rights
    e = int(st.ep[i])
    b.ep_square = None if e < 0 else e
    b.halfmove_clock = int(st.half[i])
    return b


def code_to_move(code: int) -> chess.Move:
    frm, to, promo = code & 63, (code >> 6) & 63, (code >> 12) & 3
    if promo:
        return chess.Move(frm, to, promotion=[None, chess.KNIGHT, chess.BISHOP, chess.ROOK][promo])
    return chess.Move(frm, to)


def move_to_code(move: chess.Move, board: chess.Board) -> int:
    promo = 0
    if move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        promo = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3}[move.promotion]
    return move.from_square | (move.to_square << 6) | (promo << 12)


# --------------------------------------------------------------------------- #
# Attacks
# --------------------------------------------------------------------------- #
def attacked(sq65: torch.Tensor, t: torch.Tensor, by_white: torch.Tensor, T) -> torch.Tensor:
    """Is square ``t`` attacked by the given colour? sq65 is [M, 65] (column 64 = empty)."""
    M = sq65.shape[0]
    bw = by_white[:, None]
    # knights
    pn = sq65.gather(1, T["kn"][t])
    hit = (pn == torch.where(bw, 2, 8).to(torch.int8)).any(1)
    # king
    pk = sq65.gather(1, T["kg"][t])
    hit |= (pk == torch.where(bw, 6, 12).to(torch.int8)).any(1)
    # pawns (table row 0 lists white pawns' origin squares, row 1 black pawns')
    pp = sq65.gather(1, T["pawn_att"][(~by_white).long(), t])
    hit |= (pp == torch.where(bw, 1, 7).to(torch.int8)).any(1)
    # sliders: first piece met along each ray
    rays = T["ray"][t]  # [M, 8, 7]
    pr = sq65.gather(1, rays.reshape(M, 56)).reshape(M, 8, 7)
    nz = pr != 0
    first = nz.to(torch.uint8).argmax(2)  # index of first blocker (0 if none)
    has = nz.any(2)
    fp = pr.gather(2, first[..., None]).squeeze(2)  # [M, 8]
    off = torch.where(by_white, 0, 6).to(torch.int8)[:, None]
    rook, bishop, queen = 4 + off, 3 + off, 5 + off
    orth = (fp[:, :4] == rook) | (fp[:, :4] == queen)
    diag = (fp[:, 4:] == bishop) | (fp[:, 4:] == queen)
    hit |= (has[:, :4] & orth).any(1) | (has[:, 4:] & diag).any(1)
    return hit


def _pad65(sq: torch.Tensor) -> torch.Tensor:
    return F.pad(sq, (0, 1))


def _king_square(sq: torch.Tensor, white: torch.Tensor) -> torch.Tensor:
    code = torch.where(white, 6, 12).to(torch.int8)[:, None]
    return (sq == code).to(torch.float32).argmax(1)


def in_check(st: State) -> torch.Tensor:
    T = tables(st.device)
    ksq = _king_square(st.sq, st.stm)
    return attacked(_pad65(st.sq), ksq, ~st.stm, T)


# --------------------------------------------------------------------------- #
# Move making
# --------------------------------------------------------------------------- #
def _apply_sq(sq65: torch.Tensor, frm, to, promo, ep, T):
    """Apply moves to padded boards. Returns (new_sq65, is_pawn, is_capture)."""
    M = sq65.shape[0]
    ar = torch.arange(M, device=sq65.device)
    piece = sq65[ar, frm]
    white = piece <= 6
    ptype = (piece.long() - 1) % 6 + 1
    is_pawn = ptype == 1
    tgt = sq65[ar, to]
    ff, ft = frm % 8, to % 8
    ep_cap = is_pawn & (to == ep) & (ff != ft) & (ep >= 0)
    cap_sq = torch.where(ep_cap, torch.where(white, to - 8, to + 8), torch.full_like(to, 64))
    to_rank = to // 8
    is_promo = is_pawn & ((to_rank == 7) | (to_rank == 0))
    promo_type = T["promo_piece"][promo]
    placed = torch.where(is_promo, promo_type + torch.where(white, 0, 6).to(torch.int8), piece)
    is_castle = (ptype == 6) & ((ft - ff).abs() == 2)
    ks = ft > ff
    rook_from = torch.where(is_castle, torch.where(ks, frm + 3, frm - 4), torch.full_like(to, 64))
    rook_to = torch.where(is_castle, torch.where(ks, frm + 1, frm - 1), torch.full_like(to, 64))
    rook_code = torch.where(white, 4, 10).to(torch.int8)

    new = sq65.clone()
    zero = torch.zeros((), dtype=torch.int8, device=sq65.device)  # device scalar: graph-capture safe
    new[ar, frm] = zero
    new[ar, cap_sq] = zero
    new[ar, rook_from] = zero
    new[ar, to] = placed
    new[ar, rook_to] = rook_code
    new[:, 64] = zero
    is_capture = (tgt != 0) | ep_cap
    return new, is_pawn, is_capture


def make_moves(st: State, code: torch.Tensor) -> State:
    """Apply one packed move per row (moves must be legal)."""
    T = tables(st.device)
    frm = (code & 63).long()
    to = ((code >> 6) & 63).long()
    promo = ((code >> 12) & 3).long()
    new65, is_pawn, is_cap = _apply_sq(_pad65(st.sq), frm, to, promo, st.ep, T)
    castle = st.castle & ~T["clear"][frm] & ~T["clear"][to]
    double = is_pawn & ((to - frm).abs() == 16)
    ep = torch.where(double, (frm + to) // 2, torch.full_like(to, -1))
    half = torch.where(is_pawn | is_cap, torch.zeros_like(st.half), st.half + 1)
    return State(new65[:, :64].contiguous(), ~st.stm, castle, ep, half)


# --------------------------------------------------------------------------- #
# Legal move generation
# --------------------------------------------------------------------------- #
@dataclass
class Moves:
    code: torch.Tensor      # [B, A] int32 packed moves (garbage where not valid)
    valid: torch.Tensor     # [B, A] bool
    n: torch.Tensor         # [B] number of legal moves
    in_check: torch.Tensor  # [B] bool
    pol: torch.Tensor       # [B, A] policy index of each move (0 where invalid)


OVERFLOW = [torch.zeros((), dtype=torch.long)]  # device counter of truncated positions (fast path)


def legal_moves(st: State, max_moves: int = MAX_MOVES, exact: bool = True) -> Moves:
    """All legal moves for each position.

    A fast pass checks up to ``_PSEUDO_FAST`` pseudo-legal candidates per position. With
    ``exact=True`` (default) the rare positions with more candidates are recomputed with the
    full budget (costs one host sync). ``exact=False`` never syncs: such positions are simply
    truncated and counted in ``OVERFLOW`` so callers can report it.
    """
    mv, count = _legal_moves_k(st, max_moves, _PSEUDO_FAST)
    over = count > _PSEUDO_FAST
    if not exact:
        if OVERFLOW[0].device != st.device:
            OVERFLOW[0] = OVERFLOW[0].to(st.device)
        OVERFLOW[0] += over.sum()
        return mv
    if bool(over.any()):
        idx = over.nonzero().squeeze(1)
        full, _ = _legal_moves_k(st.index(idx), max_moves, _PSEUDO_FULL)
        mv.code[idx] = full.code
        mv.valid[idx] = full.valid
        mv.n[idx] = full.n
        mv.pol[idx] = full.pol
    return mv


def _legal_moves_k(st: State, max_moves: int, K: int) -> Tuple[Moves, torch.Tensor]:
    T = tables(st.device)
    B = len(st)
    dev = st.device
    sq65 = _pad65(st.sq)                      # [B, 65] int8
    white = st.stm
    piece = st.sq                             # [B, 64]
    is_w = (piece >= 1) & (piece <= 6)
    is_b = piece >= 7
    own = torch.where(white[:, None], is_w, is_b)
    enemy = torch.where(white[:, None], is_b, is_w)
    own65 = F.pad(own, (0, 1))
    enemy65 = F.pad(enemy, (0, 1))
    empty65 = sq65 == 0
    ptype = torch.where(piece > 0, (piece.long() - 1) % 6 + 1, torch.zeros_like(piece, dtype=torch.long))

    # --- sliders: [B, 64, 8, 7] ------------------------------------------------
    ray = T["ray"]                                                    # [64, 8, 7]
    ray_flat = ray.reshape(-1)
    occ_ray = (~empty65[:, ray_flat]).reshape(B, 64, 8, 7)
    occ_i = occ_ray.to(torch.int32)
    blockers_before = occ_i.cumsum(3) - occ_i
    own_ray = own65[:, ray_flat].reshape(B, 64, 8, 7)
    ray_ok = (blockers_before == 0) & ~own_ray & (ray < 64)[None]
    is_rook, is_bish, is_queen = ptype == 4, ptype == 3, ptype == 5
    slider_dir = torch.cat([
        ((is_rook | is_queen) & own)[:, :, None].expand(B, 64, 4),
        ((is_bish | is_queen) & own)[:, :, None].expand(B, 64, 4),
    ], dim=2)                                                        # [B, 64, 8]
    ray_ok = (ray_ok & slider_dir[..., None]).reshape(B, 64, 56)

    # --- knights / kings -----------------------------------------------------------
    kn, kg = T["kn"], T["kg"]
    kn_ok = (~own65[:, kn.reshape(-1)]).reshape(B, 64, 8) & (kn < 64)[None] & ((ptype == 2) & own)[:, :, None]
    kg_ok = (~own65[:, kg.reshape(-1)]).reshape(B, 64, 8) & (kg < 64)[None] & ((ptype == 6) & own)[:, :, None]

    # --- pawns ----------------------------------------------------------------------
    col = (~white).long()
    ptgt = T["pawn"][col]                                            # [B, 64, 4]
    tflat = ptgt.reshape(B, 256)
    e_at = empty65.gather(1, tflat).reshape(B, 64, 4)
    en_at = enemy65.gather(1, tflat).reshape(B, 64, 4)
    mid = ptgt[:, :, 0]
    mid_empty = empty65.gather(1, mid)
    ep_hit = (ptgt == st.ep[:, None, None]) & (st.ep >= 0)[:, None, None]
    is_pawn_own = (ptype == 1) & own
    p_valid = torch.stack([
        e_at[..., 0],
        e_at[..., 1] & mid_empty,
        en_at[..., 2] | ep_hit[..., 2],
        en_at[..., 3] | ep_hit[..., 3],
    ], dim=2) & (ptgt < 64) & is_pawn_own[:, :, None]

    # --- castling ---------------------------------------------------------------------
    home = torch.where(white, 0, 56)                                 # base square of home rank
    e_sq, f_sq, d_sq = home + 4, home + 5, home + 3
    chk = attacked(
        sq65.repeat_interleave(3, 0),
        torch.stack([e_sq, f_sq, d_sq], dim=1).reshape(-1),
        (~white).repeat_interleave(3), T,
    ).reshape(B, 3)
    e_at_sq = sq65.gather(1, e_sq[:, None]).squeeze(1)
    king_home = e_at_sq == torch.where(white, 6, 12).to(torch.int8)
    em = lambda off: empty65.gather(1, (home + off)[:, None]).squeeze(1)
    ks_ok = torch.where(white, st.castle[:, 0], st.castle[:, 2]) & king_home & em(5) & em(6) & ~chk[:, 0] & ~chk[:, 1]
    qs_ok = torch.where(white, st.castle[:, 1], st.castle[:, 3]) & king_home & em(1) & em(2) & em(3) & ~chk[:, 0] & ~chk[:, 2]
    c_valid = torch.zeros(B, 64, 2, dtype=torch.bool, device=dev)
    ar_b = torch.arange(B, device=dev)
    c_valid[ar_b, e_sq] = torch.stack([ks_ok, qs_ok], dim=1)

    cand = torch.cat([ray_ok, kn_ok, kg_ok, p_valid, c_valid], dim=2).reshape(B, 64 * _NSLOT)

    # --- compact the pseudo-legal candidates ----------------------------------------------
    count = cand.sum(1)
    vals, idx = torch.topk(cand.to(torch.float32), K, dim=1)
    cvalid = vals > 0
    frm = idx // _NSLOT
    slot = idx % _NSLOT
    slot_to = T["slot_to"][col]                                      # [B, 64, 78]
    to = slot_to.reshape(B, 64 * _NSLOT).gather(1, idx)
    to = torch.where(cvalid, to, torch.zeros_like(to))

    # --- legality: make each move and test our king --------------------------------------
    R = B * K
    sq_r = sq65.repeat_interleave(K, 0)
    ep_r = st.ep.repeat_interleave(K, 0)
    frm_r, to_r = frm.reshape(R), to.reshape(R)
    new65, _, _ = _apply_sq(sq_r, frm_r, to_r, torch.zeros_like(frm_r), ep_r, T)
    w_r = white.repeat_interleave(K, 0)
    ksq = _king_square(new65[:, :64], w_r)
    illegal = attacked(new65, ksq, ~w_r, T).reshape(B, K)
    lvalid = cvalid & ~illegal

    # --- promotion expansion + final compaction -----------------------------------------------
    to_rank = to // 8
    fp = piece.gather(1, frm)
    is_promo = lvalid & (((fp.long() - 1) % 6 + 1) == 1) & ((to_rank == 0) | (to_rank == 7))
    base = (frm | (to << 6)).to(torch.int32)
    codes = torch.cat([base, base | (1 << 12), base | (2 << 12), base | (3 << 12)], dim=1)   # [B, 4K]
    valid4 = torch.cat([lvalid, is_promo, is_promo, is_promo], dim=1)
    vals2, idx2 = torch.topk(valid4.to(torch.float32), max_moves, dim=1)
    valid = vals2 > 0
    code = codes.gather(1, idx2)
    code = torch.where(valid, code, torch.zeros_like(code))
    n = valid.sum(1)

    ksq0 = _king_square(st.sq, white)
    chk_now = attacked(sq65, ksq0, ~white, T)
    lk = T["lookup"]
    pol = lk[((code >> 12) & 3).long(), (code & 4095).long()]
    pol = torch.where(valid, pol, torch.zeros_like(pol))
    return Moves(code, valid, n, chk_now, pol), count


# --------------------------------------------------------------------------- #
# Terminal detection, encoding, hashing
# --------------------------------------------------------------------------- #
def insufficient_material(sq: torch.Tensor) -> torch.Tensor:
    """Same rule as ``chess.Board.is_insufficient_material`` (both sides)."""
    B = sq.shape[0]
    dev = sq.device
    t = torch.arange(64, device=dev)
    dark = (((t // 8) + (t % 8)) % 2 == 0)  # a1 is dark

    def count(code):
        return (sq == code).sum(1)

    def side(off: int) -> Tuple[torch.Tensor, ...]:
        return (count(1 + off), count(2 + off), count(3 + off), count(4 + off), count(5 + off))

    wp, wn, wb, wr, wq = side(0)
    bp, bn, bb, br, bq = side(6)
    pawns = wp + bp
    knights = wn + bn
    bishops_mask = (sq == 3) | (sq == 9)
    any_dark_b = (bishops_mask & dark[None]).any(1)
    any_light_b = (bishops_mask & ~dark[None]).any(1)
    same_color_bishops = ~(any_dark_b & any_light_b)

    def insuff(p, n, b, r, q, total, opp_total_nonking_nonqueen):
        heavy = (p + r + q) > 0
        knight_case = (n > 0) & (total <= 2) & (opp_total_nonking_nonqueen == 0)
        bishop_case = (n == 0) & (b > 0) & same_color_bishops & (pawns == 0) & (knights == 0)
        pure_king = (n == 0) & (b == 0)
        return ~heavy & torch.where(n > 0, knight_case, torch.where(b > 0, bishop_case, pure_king))

    w_total = wp + wn + wb + wr + wq + 1
    b_total = bp + bn + bb + br + bq + 1
    w_opp = bp + bn + bb + br      # opponent pieces except king and queens
    b_opp = wp + wn + wb + wr
    return insuff(wp, wn, wb, wr, wq, w_total, w_opp) & insuff(bp, bn, bb, br, bq, b_total, b_opp)


def terminal_info(st: State, mv: Moves) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (is_terminal [B] bool, value [B] float from side-to-move's view)."""
    no_moves = mv.n == 0
    mate = no_moves & mv.in_check
    draw = (no_moves & ~mv.in_check) | insufficient_material(st.sq) | (st.half >= 100)
    is_term = mate | draw
    value = torch.where(mate, -1.0, 0.0)
    return is_term, value


def to_planes(st: State) -> torch.Tensor:
    """[B, 18, 8, 8] float32 planes identical to engine.encoding.board_to_planes."""
    B = len(st)
    dev = st.device
    oh = F.one_hot(st.sq.long(), 13)[..., 1:].to(torch.float32)      # [B, 64, 12]
    planes = torch.zeros(B, 18, 64, device=dev)
    planes[:, :12] = oh.permute(0, 2, 1)
    planes[:, 12:16] = st.castle.to(torch.float32)[:, :, None]
    ep_oh = torch.zeros(B, 65, device=dev)
    ep_oh.scatter_(1, torch.where(st.ep >= 0, st.ep, torch.full_like(st.ep, 64))[:, None], 1.0)
    planes[:, 16] = ep_oh[:, :64]
    planes[:, 17] = st.stm.to(torch.float32)[:, None]
    return planes.reshape(B, 18, 8, 8)


def hash_state(st: State) -> torch.Tensor:
    """64-bit position hash (wraparound int64 sum of random keys)."""
    T = tables(st.device)
    h = T["zob_sq"][T["arange64"][None, :].expand(len(st), 64), st.sq.long()].sum(1)
    h = h + (T["zob_castle"][None, :] * st.castle.to(torch.int64)).sum(1)
    h = h + T["zob_ep"][torch.where(st.ep >= 0, st.ep, torch.full_like(st.ep, 64))]
    h = h + T["zob_stm"][0] * st.stm.to(torch.int64)
    return h
