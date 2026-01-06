from players.player import Player
from game.logic import TicTacToe
import random


class RandomPlayer(Player):
    """ A agent that plays randomly """
    
    def get_move(self, game: TicTacToe):
        """ Method to select a move for the agent. """
        moves = game.get_legal_moves()
        if moves: 
            return random.choice(moves)
        return None