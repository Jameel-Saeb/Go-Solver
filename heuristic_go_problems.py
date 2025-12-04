from typing import Optional, Any, Union
import numpy as np
import torch
from go_search_problem import GoState, HeuristicGoProblem
BLACK = 0
WHITE = 1


class GoProblemSimpleHeuristic(HeuristicGoProblem):
    def __init__(self, size: int = 5, state=None, player_to_move: int = 0):
        super().__init__(size=size, state=state, player_to_move=player_to_move)

    def heuristic(self, state, player_index):
        """
        Very simple heuristic that just compares the number of pieces for each player
        
        Having more pieces than the opponent means that some were captured, capturing is generally good.
        Returns value from BLACK's perspective: positive = good for BLACK, negative = good for WHITE.
        """
        black_stones = len(state.get_pieces_coordinates(BLACK))
        white_stones = len(state.get_pieces_coordinates(WHITE))

        return black_stones - white_stones

    def __str__(self) -> str:
        return "Simple Heuristic"


class GoProblemLearnedHeuristic(HeuristicGoProblem):
    def __init__(self, model=None, size: int = 5, state=None, player_to_move: int = 0):
        super().__init__(size=size, state=state, player_to_move=player_to_move)
        self.model = model

    def encoding(self, state: GoState):
        board_size = state.size
        black = np.array(state.get_pieces_array(0), dtype=float)
        white = np.array(state.get_pieces_array(1), dtype=float)
        empty = 1.0 - (black + white)
        empty = np.clip(empty, 0.0, 1.0)
        # player_to_move() method returns 0 for BLACK, 1 for WHITE
        if state.player_to_move() == 0:
            to_move = np.ones((board_size, board_size), dtype=float)
        else:
            to_move = np.zeros((board_size, board_size), dtype=float)

        features = np.stack([black, white, empty, to_move], axis=0)
        return features.flatten()

    def heuristic(self, state: GoState, player_index: int):
        if self.model is None:
            return 0.0

        feats = np.array(self.encoding(state), dtype=np.float32)
        x = torch.tensor(feats, dtype=torch.float32)
        with torch.no_grad():
            self.model.eval()
            if x.dim() == 1:
                x = x.unsqueeze(0)
            pred = self.model(x).squeeze().item()
        return float(pred)

    def __str__(self) -> str:
        return "Learned Heuristic"
