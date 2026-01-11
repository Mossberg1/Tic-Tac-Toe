# Import Modules
from players.player import Player
from game.symbol import Symbol
from game.logic import TicTacToe
import pickle
import numpy as np
import random

# Perfect Strategy Player using precomputed Q-Table
class PerfectStrategyPlayer(Player):
    def __init__(self, symbol: Symbol):
        super().__init__(symbol)

        # Load the perfect strategy Q-table from file
        with open('models/perfect_policy.pkl', 'rb') as f:
            self._q_table = pickle.load(f)
    
    def get_move(self, game: TicTacToe):         
        state = game.get_board_state()
        valid_moves = game.get_legal_moves()

        try:
            # Get Q-values for the current state
            q_values = self._q_table[state]
            # Initialize best move and value
            best_move = 0
            best_value = q_values[0] # Grab the first value in the list
            
            # Loop through all valid moves
            for i in range(1, len(valid_moves)):
                value = q_values[i]

                # If this move has a higher Q-value, update best move
                if value > best_value:
                    best_value = value
                    best_move = i
            
            # Return the best move found in valid moves
            return valid_moves[best_move]
        
        except KeyError:
            # If state not found in Q-table, return a random valid move
            return random.choice(valid_moves) if valid_moves else None
        
                
