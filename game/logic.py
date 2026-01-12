from typing import List, Optional, Tuple

class TicTacToe:
    """Core game logic for Tic Tac Toe"""

    def __init__(self) -> None:
        self.board: List[List[int]] = [[0 for _ in range(3)] for _ in range(3)]
        self.current_player: int = 1
        self.game_over: bool = False
        self.winner: Optional[int] = None
    
    def make_move(self, row: int, col: int) -> bool:
        """Make a move on the board. Returns True if successful, False otherwise."""
        if not self.is_valid_move(row, col):
            return False

        self.board[row][col] = self.current_player

        # Check for winner
        if self.check_winner(self.current_player):
            self.game_over = True
            self.winner = self.current_player
        elif self.is_board_full():
            self.game_over = True
            self.winner = 0  # Draw
        else:
            self.current_player = 3 - self.current_player  # Switch players

        return True

    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if a move is valid"""
        if self.game_over:
            return False
        if row < 0 or row >= 3 or col < 0 or col >= 3: # Check if move is inside the board.
            return False
        
        return self.board[row][col] == 0 # Check if position is empty

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Return list of all legal moves as (row, col) tuples"""
        moves: List[Tuple[int, int]] = []
        for row in range(3):
            for col in range(3):
                # Append only empty positions
                if self.board[row][col] == 0:
                    moves.append((row, col))
        return moves

    def is_board_full(self) -> bool:
        """Check if the board is completely filled"""
        for row in range(3):
            for col in range(3):
                # If i empty position is found break early. 
                if self.board[row][col] == 0:
                    return False
        return True

    def check_winner(self, player: int) -> bool:
        """Check if the specified player has won"""
        # Check rows
        for row in range(3):
            if all(self.board[row][col] == player for col in range(3)):
                return True
        
        # Check columns
        for col in range(3):
            if all(self.board[row][col] == player for row in range(3)):
                return True
        
        # Check diagonals
        if all(self.board[i][i] == player for i in range(3)):
            return True
        if all(self.board[i][2-i] == player for i in range(3)):
            return True
        
        return False
    
    def get_board_state(self) -> Tuple[int, ...]:
        """Return board state"""
        return tuple(cell for row in self.board for cell in row)

    def reset(self) -> None:
        """Reset the game to initial state"""
        self.board = [[0 for _ in range(3)] for _ in range(3)]
        self.current_player = 1
        self.game_over = False
        self.winner = None

    def __str__(self) -> str:
        """String representation of the board for debugging"""
        symbols = {0: '.', 1: 'O', 2: 'X'}
        lines = []
        for row in self.board:
            lines.append(' '.join(symbols[cell] for cell in row))
        return '\n'.join(lines)