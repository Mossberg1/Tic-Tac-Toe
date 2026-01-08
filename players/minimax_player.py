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
        #self.transposition_table.clear() 
        legal_moves = game.get_legal_moves()
        
        if not legal_moves:
            return None
        
        best_move = None
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf
        
        for move in legal_moves:
            # Simulate the move
            row, col = move
            game.board[row][col] = self.symbol
            
            # Get the minimax value
            value = self.minimax(game, 0, False, alpha, beta)
            
            # Undo the move
            game.board[row][col] = 0
            
            if value > best_value:
                best_value = value
                best_move = move
            
            alpha = max(alpha, value)
        
        return best_move
    

    def minimax(self, game: 'TicTacToe', depth: int, is_maximizing: bool, alpha: float, beta: float) -> float:
            # Save original alpha to determine flag later
            alpha_orig = alpha

            # 1. Transposition Table Lookup
            board_state = game.get_board_state()

            # Key: In chess, often just board_hash is enough,
            # but here we include is_maximizing for safety.
            cache_key = (board_state, is_maximizing)

            if cache_key in self.transposition_table:
                tt_score, tt_flag, tt_depth = self.transposition_table[cache_key]

                # In chess we check: if tt_depth >= remaining_depth
                # Here we keep it simple: If we found this at the same depth or deeper (more exact)
                if tt_depth >= depth:  # Or just ignore depth check for simple TTT
                    if tt_flag == FLAG_EXACT:
                        return tt_score
                    elif tt_flag == FLAG_LOWERBOUND:
                        alpha = max(alpha, tt_score)
                    elif tt_flag == FLAG_UPPERBOUND:
                        beta = min(beta, tt_score)

                    if alpha >= beta:
                        return tt_score

            self.nodes_explored += 1

            # 2. Terminal states
            if game.check_winner(self.symbol):
                return 10 - depth
            opponent = 3 - self.symbol
            if game.check_winner(opponent):
                return -10 + depth
            if game.is_board_full():
                return 0

            # 3. Depth limit (Quiescence search would be here in chess)
            if self.depth_limit is not None and depth >= self.depth_limit:
                return self.evaluate_position(game)

            # 4. Recursive search
            legal_moves = game.get_legal_moves()
            
            best_score = -math.inf if is_maximizing else math.inf
            
            if is_maximizing:
                for move in legal_moves:
                    row, col = move
                    game.board[row][col] = self.symbol
                    
                    score = self.minimax(game, depth + 1, False, alpha, beta)
                    
                    game.board[row][col] = 0
                    
                    best_score = max(best_score, score)
                    alpha = max(alpha, best_score)
                    
                    if beta <= alpha:
                        break # Beta cutoff
            else:
                for move in legal_moves:
                    row, col = move
                    game.board[row][col] = opponent
                    
                    score = self.minimax(game, depth + 1, True, alpha, beta)
                    
                    game.board[row][col] = 0
                    
                    best_score = min(best_score, score)
                    beta = min(beta, best_score)
                    
                    if beta <= alpha:
                        break # Alpha cutoff

            # 5. Save to Transposition Table
            # Now we determine which flag to set
            tt_flag = FLAG_EXACT
            if best_score < alpha_orig:
                tt_flag = FLAG_UPPERBOUND  # Fail-low: We didn't find anything better than we already had
            elif best_score >= beta:
                tt_flag = FLAG_LOWERBOUND  # Fail-high: We caused a cutoff
                
            self.transposition_table[cache_key] = (best_score, tt_flag, depth)
            
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