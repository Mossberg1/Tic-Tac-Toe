from players.player import Player
from game.symbol import Symbol
from game.logic import TicTacToe
import pickle
import numpy as np
import random

class PerfectStrategyPlayer(Player):
    def __init__(self, symbol: Symbol):
        super().__init__(symbol)
        with open('models/perfect_policy.pkl', 'rb') as f:
            self._q_table = pickle.load(f)
    
    def get_move(self, game: TicTacToe):         
        state = game.get_board_state()
        valid_moves = game.get_legal_moves()

        try:
            q_values = self._q_table[state]
            best_move = 0
            best_value = q_values[0]
            
            for i in range(1, len(valid_moves)):
                value = q_values[i]
                if value > best_value:
                    best_value = value
                    best_move = i
            
            return valid_moves[best_move]
        except KeyError:
            return random.choice(valid_moves) if valid_moves else None
        
        """
        state = game.get_board_state()
        valid_moves = game.get_legal_moves()

        try:
            action_probs = self._q_table[state]
            chosen_relative_index = np.argmax(action_probs)
            chosen_absolute_index = valid_moves[chosen_relative_index]
            
            return (chosen_absolute_index % 3, chosen_absolute_index // 3)
        except KeyError:
            return random.choice(valid_moves) if valid_moves else None
        """
                
