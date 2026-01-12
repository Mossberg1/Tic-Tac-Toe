
from players.player import Player
from game.logic import TicTacToe
from game.winner_state import WinnerState
from game.symbol import Symbol
import matplotlib.pyplot as plt


class GameSimulator:
    """ Simulator class to simulate games against agents. """
    
    def __init__(self, player1: Player, player2: Player, n_simulations: int, player_to_track: Symbol):
        if player1.symbol == player2.symbol: # Player cant have the same symbol.
            raise ValueError()
        
        self.game = TicTacToe()
        self.player1 = player1
        self.player2 = player2
        self.n_simulations = n_simulations
        
        # Properties to plot after simulation.
        self._tracked_player = player_to_track
        self._n_wins = 0
        self._n_draws = 0
        self._n_losses = 0
        
        
    def simulate(self):
        """ Method to simulate n number of games. """
        for i in range(self.n_simulations):
            while self.game.game_over == False: 
                move = None
                
                if self.game.current_player == self.player1.symbol: # Player 1s turn.
                    move = self.player1.get_move(self.game)
                else: # Is player 2s turn.
                    move = self.player2.get_move(self.game)
                
                # Make the move.
                if move is not None:
                    row, col = move
                    self.game.make_move(row, col)
                else: # Should never happen because if move is None game should be over. 
                    raise ValueError("Error: Move was None")
             
            # Get the winner.   
            match (self.game.winner):
                case self._tracked_player: # Check if the tracked player has won. 
                    self._n_wins += 1
                case WinnerState.DRAW:
                    self._n_draws += 1
                case _ :
                    self._n_losses += 1    

            self.game.reset()  

        
    
    def plot(self):
        """ Method to plot wins, draws and losses for the tracked agent. """
        categories = ['Wins', 'Draws', 'Losses']
        values = [self._n_wins, self._n_draws, self._n_losses]
        
        plt.figure(figsize=(8, 5))
        plt.bar(categories, values, color=['green', 'blue', 'red'])
        
        tracked_player = self.player1 if self.player1.symbol != self._tracked_player else self.player2
        
        plt.title(f'Game Results against: {type(tracked_player).__name__}')
        plt.ylabel('Count')
        
        for i, v in enumerate(values):
            plt.text(i, v, str(v), ha='center', va='bottom')

        plt.show()
        

