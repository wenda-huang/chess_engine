"""PUCT Monte Carlo Tree Search guided by the policy+value network.

Performance features:
  - Batched leaf evaluation: collect N leaves per round, one GPU forward pass.
  - Virtual loss: inflates visit counts on in-flight paths so batch selections
    diversify instead of piling onto the same line.
  - NN eval cache: Zobrist-keyed cache avoids re-evaluating transpositions.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np
import torch

from engine.encoding import board_to_planes, legal_move_indices
from engine.inference import InferenceRunner


class Node:
    __slots__ = ("prior", "N", "W", "virtual_loss", "children", "is_expanded")

    def __init__(self, prior: float):
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.virtual_loss = 0
        self.children: Dict[chess.Move, Node] = {}
        self.is_expanded = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


Leaf = Tuple[Node, chess.Board, List[Node]]


class NNEvalCache:
    """Zobrist-keyed cache of (priors, value) network outputs."""

    def __init__(self, max_size: int = 50_000):
        self.max_size = max_size
        self._data: OrderedDict[int, Tuple[Dict[chess.Move, float], float]] = OrderedDict()

    def get(self, board: chess.Board) -> Optional[Tuple[Dict[chess.Move, float], float]]:
        key = board._transposition_key()
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, board: chess.Board, priors: Dict[chess.Move, float], value: float) -> None:
        key = board._transposition_key()
        if key in self._data:
            self._data.move_to_end(key)
        else:
            if len(self._data) >= self.max_size:
                self._data.popitem(last=False)
            self._data[key] = (priors, value)


def _priors_from_logits(logits: np.ndarray, board: chess.Board) -> Dict[chess.Move, float]:
    moves, idxs = legal_move_indices(board)
    if not moves:
        return {}
    masked = logits[idxs]
    masked = masked - np.max(masked)
    exp = np.exp(masked)
    probs = exp / np.sum(exp)
    return {m: float(p) for m, p in zip(moves, probs)}


@torch.no_grad()
def net_eval_batch(
    boards: List[chess.Board],
    runner: InferenceRunner,
    cache: Optional[NNEvalCache] = None,
) -> List[Tuple[Dict[chess.Move, float], float]]:
    """Evaluate a batch of boards; uses cache hits and batches misses together."""
    if not boards:
        return []

    results: List[Optional[Tuple[Dict[chess.Move, float], float]]] = [None] * len(boards)
    miss_boards: List[chess.Board] = []
    miss_idx: List[int] = []

    for i, board in enumerate(boards):
        if cache is not None:
            hit = cache.get(board)
            if hit is not None:
                results[i] = hit
                continue
        miss_boards.append(board)
        miss_idx.append(i)

    if miss_boards:
        planes = np.stack([board_to_planes(b) for b in miss_boards]).astype(np.float32)
        logits_np, values_np = runner.eval_batch(planes)

        for j, board in enumerate(miss_boards):
            priors = _priors_from_logits(logits_np[j], board)
            value = float(values_np[j])
            results[miss_idx[j]] = (priors, value)
            if cache is not None:
                cache.put(board, priors, value)

    return [r for r in results if r is not None]


@torch.no_grad()
def net_eval(
    board: chess.Board,
    runner: InferenceRunner,
    cache: Optional[NNEvalCache] = None,
) -> Tuple[Dict[chess.Move, float], float]:
    """Single-position eval (delegates to the batched path)."""
    priors, value = net_eval_batch([board], runner, cache=cache)[0]
    return priors, value


def _terminal_value(board: chess.Board) -> float:
    if board.is_checkmate():
        return -1.0
    return 0.0


class MCTS:
    def __init__(
        self,
        runner: InferenceRunner,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        batch_size: int = 32,
        virtual_loss: int = 3,
        nn_cache_size: int = 50_000,
    ):
        self.runner = runner
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = max(1, batch_size)
        self.virtual_loss = virtual_loss
        self.cache = NNEvalCache(nn_cache_size) if nn_cache_size > 0 else None

    def _expand_with_result(
        self, node: Node, board: chess.Board, priors: Dict[chess.Move, float], value: float
    ) -> float:
        if board.is_game_over(claim_draw=False):
            node.is_expanded = True
            return _terminal_value(board)
        for move, p in priors.items():
            node.children[move] = Node(p)
        node.is_expanded = True
        return value

    def _add_dirichlet_noise(self, node: Node) -> None:
        moves = list(node.children.keys())
        if not moves:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
        eps = self.dirichlet_epsilon
        for move, n in zip(moves, noise):
            child = node.children[move]
            child.prior = (1 - eps) * child.prior + eps * float(n)

    def _select_child(self, node: Node) -> Tuple[chess.Move, Node]:
        best_score = -float("inf")
        best_move: Optional[chess.Move] = None
        best_child: Optional[Node] = None
        sqrt_parent = math.sqrt(node.N + node.virtual_loss + 1)
        for move, child in node.children.items():
            q = -child.Q
            n_eff = child.N + child.virtual_loss
            u = self.c_puct * child.prior * sqrt_parent / (1 + n_eff)
            score = q + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        assert best_move is not None and best_child is not None
        return best_move, best_child

    def _select_leaf(self, root: Node, board: chess.Board) -> Leaf:
        node = root
        sim_board = board.copy()
        path = [node]
        while node.is_expanded and node.children:
            move, node = self._select_child(node)
            sim_board.push(move)
            path.append(node)
        return node, sim_board, path

    def _apply_virtual_loss(self, path: List[Node]) -> None:
        for n in path:
            n.virtual_loss += self.virtual_loss

    def _clear_virtual_loss(self, path: List[Node]) -> None:
        for n in path:
            n.virtual_loss -= self.virtual_loss

    def _backprop(self, path: List[Node], value: float) -> None:
        for n in reversed(path):
            n.N += 1
            n.W += value
            value = -value

    def run(self, board: chess.Board, simulations: int, add_noise: bool = False) -> Node:
        root = Node(0.0)

        # Expand root (single position; seeds the tree).
        if board.is_game_over(claim_draw=False):
            root.is_expanded = True
        else:
            priors, value = net_eval(board, self.runner, cache=self.cache)
            self._expand_with_result(root, board, priors, value)
            if add_noise:
                self._add_dirichlet_noise(root)

        sims_done = 0
        while sims_done < simulations:
            n_batch = min(self.batch_size, simulations - sims_done)

            leaves: List[Leaf] = []
            for _ in range(n_batch):
                leaf = self._select_leaf(root, board)
                self._apply_virtual_loss(leaf[2])
                leaves.append(leaf)

            # Terminal leaves skip the network.
            to_eval: List[Tuple[int, chess.Board]] = []
            for i, (node, sim_board, path) in enumerate(leaves):
                if sim_board.is_game_over(claim_draw=False):
                    value = _terminal_value(sim_board)
                    node.is_expanded = True
                    self._clear_virtual_loss(path)
                    self._backprop(path, value)
                else:
                    to_eval.append((i, sim_board))

            if to_eval:
                boards = [b for _, b in to_eval]
                results = net_eval_batch(boards, self.runner, cache=self.cache)
                for (i, _), (priors, value) in zip(to_eval, results):
                    node, sim_board, path = leaves[i]
                    value = self._expand_with_result(node, sim_board, priors, value)
                    self._clear_virtual_loss(path)
                    self._backprop(path, value)

            sims_done += n_batch

        return root
