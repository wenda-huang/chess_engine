"""PUCT Monte Carlo Tree Search guided by the policy+value network."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np
import torch

from engine.encoding import board_to_planes, legal_move_indices
from engine.model import ChessNet


class Node:
    __slots__ = ("prior", "N", "W", "children", "is_expanded")

    def __init__(self, prior: float):
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.children: Dict[chess.Move, "Node"] = {}
        self.is_expanded = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


@torch.no_grad()
def net_eval(board: chess.Board, model: ChessNet, device: str) -> Tuple[Dict[chess.Move, float], float]:
    """Return (priors over legal moves, value) for ``board`` from the net.

    ``value`` is from the perspective of the side to move at ``board``.
    """
    planes = board_to_planes(board)
    x = torch.from_numpy(planes).unsqueeze(0).to(device)
    logits, value = model(x)
    logits = logits[0].detach().cpu().numpy()
    value = float(value.item())

    moves, idxs = legal_move_indices(board)
    if not moves:
        return {}, value
    masked = logits[idxs]
    masked = masked - np.max(masked)
    exp = np.exp(masked)
    probs = exp / np.sum(exp)
    return {m: float(p) for m, p in zip(moves, probs)}, value


def _terminal_value(board: chess.Board) -> float:
    if board.is_checkmate():
        return -1.0
    return 0.0


class MCTS:
    def __init__(
        self,
        model: ChessNet,
        device: str = "cpu",
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def _expand(self, node: Node, board: chess.Board) -> float:
        if board.is_game_over(claim_draw=False):
            node.is_expanded = True
            return _terminal_value(board)
        priors, value = net_eval(board, self.model, self.device)
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
        sqrt_parent = math.sqrt(node.N + 1)
        for move, child in node.children.items():
            q = -child.Q  # child value is from the opponent's perspective
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.N)
            score = q + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

    def run(self, board: chess.Board, simulations: int, add_noise: bool = False) -> Node:
        root = Node(0.0)
        self._expand(root, board)
        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(simulations):
            node = root
            sim_board = board.copy()
            path = [node]
            while node.is_expanded and node.children:
                move, node = self._select_child(node)
                sim_board.push(move)
                path.append(node)

            value = self._expand(node, sim_board)

            for n in reversed(path):
                n.N += 1
                n.W += value
                value = -value

        return root
