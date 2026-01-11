# Import Modules
from typing import Optional, Tuple, Dict, List, TYPE_CHECKING
from players.player import Player
from game.logic import TicTacToe
from game.symbol import Symbol
import math

# Flags for Transposition Table
FLAG_EXACT = 0
FLAG_LOWERBOUND = 1
FLAG_UPPERBOUND = 2

class MinimaxPlayer(Player):
    """AI using Minimax algorithm with alpha-beta pruning"""

    def __init__(self, symbol: Symbol, depth_limit: Optional[int] = None) -> None:
        """
        Initialize Minimax player

        Args:
            player_number: 1 or 2
            depth_limit: Maximum search depth (None for unlimited)
        """
        super().__init__(symbol)
        self.depth_limit: Optional[int] = depth_limit
        self.nodes_explored: int = 0
        
        # Transposition table for memoization: maps (board_state, depth, is_maximizing) -> score
        self.transposition_table: Dict[Tuple, Tuple[float, int, int]] = {}

    def get_move(self, game: 'TicTacToe') -> Optional[Tuple[int, int]]:
        """Get best move using minimax with alpha-beta pruning and memoization"""
        self.nodes_explored = 0
        
        # Clear the cache if desired, this decreases performance
        #self.transposition_table.clear() 
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
            value = self.minimax(game, 0, False, alpha, beta)
            
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
    

    def minimax(self, game: 'TicTacToe', depth: int, is_maximizing: bool, alpha: float, beta: float) -> float:
            # Save original alpha to determine flag later
            alpha_orig = alpha

            # Get current board state for transposition table key
            board_state = game.get_board_state()

            # Get cache key
            cache_key = (board_state, is_maximizing)

            # Check if we have already computed this position
            if cache_key in self.transposition_table:
                # Retrieve stored score, flag, and depth
                tt_score, tt_flag, tt_depth = self.transposition_table[cache_key]

                # Check if we found this at the same depth or deeper
                if tt_depth >= depth: 
                    # If the flag is exact, return the stored score
                    if tt_flag == FLAG_EXACT:
                        return tt_score
                    # If the flag is a lower bound, increase alpha value
                    elif tt_flag == FLAG_LOWERBOUND:
                        alpha = max(alpha, tt_score)
                    # If the flag is an upper bound, decrease beta value
                    elif tt_flag == FLAG_UPPERBOUND:
                        beta = min(beta, tt_score)

                    # If alpha is greater or equal to beta, return the stored score
                    if alpha >= beta:
                        return tt_score

            # Increase nodes explored
            self.nodes_explored += 1

            # Terminal states checks
            if game.check_winner(self.symbol):
                # If the agent has won, return a higher score 
                # This makes winning sooner better
                return 10 - depth
            
            # Switch to opponent symbol
            opponent = 3 - self.symbol

            if game.check_winner(opponent):
                # If the opponent has won, return a lower score
                # This makes losing later better
                return -10 + depth
            
            if game.is_board_full():
                # If it's a draw just return 0
                return 0

            # Depth limit check
            if self.depth_limit is not None and depth >= self.depth_limit:
                # Return heuristic evaluation for non-terminal positions
                return self.evaluate_position(game)

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
                    score = self.minimax(game, depth + 1, False, alpha, beta)
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
                    score = self.minimax(game, depth + 1, True, alpha, beta)
                    # Revert the move
                    game.board[row][col] = 0
                    
                    # Update best score and beta
                    best_score = min(best_score, score)
                    beta = min(beta, best_score)
                    
                    if beta <= alpha:
                        break # Alpha cutoff

            # Update flag
            tt_flag = FLAG_EXACT
            if best_score < alpha_orig:
                # Fail-low: We didn't find anything better than we already had
                # We found a score worse than the original alpha
                tt_flag = FLAG_UPPERBOUND  
            elif best_score >= beta:
                # Fail-high: We caused a cutoff
                # We found a score better than the original beta
                tt_flag = FLAG_LOWERBOUND
                
            # Save Transposition Table
            self.transposition_table[cache_key] = (best_score, tt_flag, depth)
            
            # Return the best score found
            return best_score 

    def evaluate_position(self, game: 'TicTacToe') -> float:
        """
        Heuristic evaluation for non-terminal positions
        Used when depth limit is reached
        """
        # Simple heuristic: count potential winning lines
        score = 0
        opponent = 3 - self.symbol
        
        # Check all lines (rows, cols, diagonals)
        lines = []
        
        # Rows and columns
        for i in range(3):
            lines.append([game.board[i][j] for j in range(3)])
            lines.append([game.board[j][i] for j in range(3)])
        
        # Diagonals
        lines.append([game.board[i][i] for i in range(3)])
        lines.append([game.board[i][2-i] for i in range(3)])
        
        for line in lines:
            score += self.evaluate_line(line, self.symbol, opponent)
        
        return score
    
    def evaluate_line(self, line: List[int], player: int, opponent: int) -> float:
        """Evaluate a single line (row, column, or diagonal)"""
        player_count = line.count(player)
        opponent_count = line.count(opponent)
        
        if opponent_count == 0:
            if player_count == 2:
                return 5  # About to win
            elif player_count == 1:
                return 1  # Potential
        elif player_count == 0:
            if opponent_count == 2:
                return -5  # Opponent about to win
            elif opponent_count == 1:
                return -1  # Opponent potential
        
        return 0  # Line blocked by both players