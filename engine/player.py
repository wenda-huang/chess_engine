"""High-level move selection and position evaluation on top of MCTS."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import chess
import numpy as np

from engine.config import Config
from engine.encoding import POLICY_SIZE, move_to_index
from engine.mcts import MCTS, Node, net_eval
from engine.model import ChessNet


class EnginePlayer:
    def __init__(self, model: ChessNet, config: Optional[Config] = None):
        self.model = model
        self.config = config or Config()
        self.mcts = MCTS(
            model,
            device=self.config.device,
            c_puct=self.config.c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
        )

    def search(self, board: chess.Board, simulations: int, add_noise: bool = False) -> Node:
        return self.mcts.run(board, simulations, add_noise=add_noise)

    def select_move(
        self,
        board: chess.Board,
        simulations: Optional[int] = None,
        temperature: float = 0.0,
        add_noise: bool = False,
    ) -> Tuple[chess.Move, Node]:
        sims = simulations or self.config.mcts_simulations
        root = self.search(board, sims, add_noise=add_noise)
        moves = list(root.children.keys())
        if not moves:
            raise ValueError("No legal moves to select from")
        visits = np.array([root.children[m].N for m in moves], dtype=np.float64)

        if temperature <= 1e-6:
            move = moves[int(np.argmax(visits))]
        else:
            logits = np.power(visits, 1.0 / temperature)
            probs = logits / np.sum(logits)
            move = moves[int(np.random.choice(len(moves), p=probs))]
        return move, root

    def policy_target(self, root: Node) -> np.ndarray:
        """Visit-count distribution over the full policy vocabulary."""
        vec = np.zeros(POLICY_SIZE, dtype=np.float32)
        total = sum(child.N for child in root.children.values())
        if total == 0:
            return vec
        for move, child in root.children.items():
            vec[move_to_index(move)] = child.N / total
        return vec

    def evaluate_position(
        self, board: chess.Board, simulations: Optional[int] = None, top_k: int = 5
    ) -> Dict[str, object]:
        """Evaluate a position for the board editor / analysis UI.

        Returns value, win probability and the top-K suggested moves.
        """
        if board.is_game_over(claim_draw=False):
            outcome = board.outcome(claim_draw=False)
            value = 0.0
            if outcome is not None and outcome.winner is not None:
                value = 1.0 if outcome.winner == board.turn else -1.0
            return {
                "value": value,
                "win_prob": (value + 1) / 2,
                "game_over": True,
                "result": board.result(),
                "suggestions": [],
            }

        sims = simulations or self.config.mcts_simulations
        root = self.search(board, sims)
        # Direct net value for the position (side-to-move perspective).
        _, net_value = net_eval(board, self.model, self.config.device)

        suggestions: List[dict] = []
        moves = sorted(root.children.items(), key=lambda kv: kv[1].N, reverse=True)
        for move, child in moves[:top_k]:
            q = -child.Q  # value from the perspective of the side to move at root
            suggestions.append(
                {
                    "uci": move.uci(),
                    "san": board.san(move),
                    "visits": child.N,
                    "value": round(q, 4),
                    "win_prob": round((q + 1) / 2, 4),
                }
            )

        root_value = root.Q if root.N > 0 else net_value
        return {
            "value": round(root_value, 4),
            "net_value": round(net_value, 4),
            "win_prob": round((root_value + 1) / 2, 4),
            "game_over": False,
            "suggestions": suggestions,
        }
