from typing import Optional, Tuple, List, TYPE_CHECKING
import math
from players.player import Player

if TYPE_CHECKING:
    from game.logic import TicTacToe

class MinimaxPlayer(Player):
    """AI using basic Minimax algorithm"""

    def __init__(self, symbol, depth_limit: Optional[int] = None) -> None:
        super().__init__(symbol)
        self.depth_limit: Optional[int] = depth_limit
        self.nodes_explored: int = 0

    def get_move(self, game: 'TicTacToe') -> Optional[Tuple[int, int]]:
        """Get best move using minimax"""
        self.nodes_explored = 0
        legal_moves = game.get_legal_moves()
        
        if not legal_moves:
            return None
        
        best_move = None
        best_value = -math.inf
        
        for move in legal_moves:
            # Simulate the move
            row, col = move
            game.board[row][col] = self.symbol.value
            
            # Get the minimax value
            value = self.minimax(game, 0, False)
            
            # Undo the move
            game.board[row][col] = 0
            
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move
    
    def minimax(self, game: 'TicTacToe', depth: int, is_maximizing: bool) -> float:
        self.nodes_explored += 1
        
        # Check terminal states
        if game.check_winner(self.symbol.value):
            return 10 - depth  # Prefer faster wins
        
        opponent = 3 - self.symbol.value
        if game.check_winner(opponent):
            return -10 + depth  # Prefer slower losses
        
        if game.is_board_full():
            return 0  # Draw
        
        # Check depth limit
        if self.depth_limit is not None and depth >= self.depth_limit:
            return self.evaluate_position(game)
        
        # Recursive search
        legal_moves = game.get_legal_moves()
        
        if is_maximizing:
            best_score = -math.inf
            for move in legal_moves:
                row, col = move
                game.board[row][col] = self.symbol.value
                score = self.minimax(game, depth + 1, False)
                game.board[row][col] = 0
                best_score = max(best_score, score)
            return best_score
        else:
            best_score = math.inf
            for move in legal_moves:
                row, col = move
                game.board[row][col] = opponent
                score = self.minimax(game, depth + 1, True)
                game.board[row][col] = 0
                best_score = min(best_score, score)
            return best_score

    def evaluate_position(self, game: 'TicTacToe') -> float:
        """
        Heuristic evaluation for non-terminal positions
        Used when depth limit is reached
        """
        score = 0
        opponent = 3 - self.symbol.value
        
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
            score += self.evaluate_line(line, self.symbol.value, opponent)
        
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
