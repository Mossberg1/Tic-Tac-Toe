# Import Modules
from typing import Optional, Tuple, Dict, List, TYPE_CHECKING
from players.player import Player
from game.logic import TicTacToe
from game.symbol import Symbol
import math

class MinimaxPlayer(Player):
    """AI using Minimax algorithm with alpha-beta pruning"""

    def __init__(self, symbol: Symbol) -> None:
        super().__init__(symbol)
        self.nodes_explored = 0


    def get_move(self, game: 'TicTacToe') -> Optional[Tuple[int, int]]:
        """Get best move using minimax with alpha-beta pruning"""
        self.nodes_explored = 0
        
        legal_moves = game.get_legal_moves()
        
        if not legal_moves:
            return None
        
        # Define variables to track best move and value
        best_move = None
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf
        
        # Loop through all legal moves to find the best one
        for move in legal_moves:
            # Simulate the move
            row, col = move
            # Simulate a move on the board
            game.board[row][col] = self.symbol
            
            # Get the minimax value
            value = self.minimax(game, False, alpha, beta)
            
            # Undo the move, since it was not a real move
            game.board[row][col] = 0
            
            # Check if this move is better than the best found so far
            if value > best_value:
                best_value = value
                best_move = move
            
            # Update alpha
            alpha = max(alpha, value)
        
        # Return the best move we found
        return best_move
    

    def minimax(self, game: 'TicTacToe', is_maximizing: bool, alpha: float, beta: float) -> float:
            self.nodes_explored += 1

            # Terminal states checks
            if game.check_winner(self.symbol):
                # If the agent has won, return a higher score 
                # This makes winning sooner better
                return 1
            
            # Switch player
            opponent = 3 - self.symbol

            if game.check_winner(opponent):
                # If the opponent has won, return a lower score
                # This makes losing later better
                return -1
            
            if game.is_board_full():
                # If it's a draw just return 0
                return 0

            # Recursive search
            legal_moves = game.get_legal_moves()
            
            best_score = -math.inf if is_maximizing else math.inf
            
            if is_maximizing:
                # Attempt to maximize the score for the agent
                for move in legal_moves:
                    row, col = move
                    
                    # Simulate move
                    game.board[row][col] = self.symbol
                    # Get the minimax score from the next depth
                    score = self.minimax(game, False, alpha, beta)
                    # Revert the move
                    game.board[row][col] = 0
                    
                    # Update best score and alpha
                    best_score = max(best_score, score)
                    alpha = max(alpha, best_score)
                    
                    if beta <= alpha:
                        break # Beta cutoff
            else:
                # Attempt to minimize the score for the opponent
                for move in legal_moves:
                    row, col = move

                    # Simulate move
                    game.board[row][col] = opponent
                    # Get the minimax score from the next depth
                    score = self.minimax(game, True, alpha, beta)
                    # Revert the move
                    game.board[row][col] = 0
                    
                    # Update best score and beta
                    best_score = min(best_score, score)
                    beta = min(beta, best_score)
                    
                    if beta <= alpha:
                        break # Alpha cutoff
            
            return best_score 
