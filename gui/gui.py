import sys
from pathlib import Path

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import pygame
from game.logic import TicTacToe
from players.minimax_player import MinimaxPlayer
from players.qlearn_player import QLearnPlayer
from players.random_player import RandomPlayer
from players.perfect_strategy_player import PerfectStrategyPlayer
from game.symbol import Symbol
import argparse
import time

class TicTacToeGUI:
    
    def __init__(self, width=600, height=600):
        pygame.init()
        
        # Constants
        self.WIDTH = width
        self.HEIGHT = height
        self.LINE_WIDTH = 15
        self.BOARD_ROWS = 3
        self.BOARD_COLS = 3
        self.SQUARE_SIZE = self.WIDTH // self.BOARD_COLS
        self.CIRCLE_RADIUS = self.SQUARE_SIZE // 3
        self.CIRCLE_WIDTH = 15
        self.CROSS_WIDTH = 25
        self.SPACE = self.SQUARE_SIZE // 4
        
        # Colors
        self.BG_COLOR = (28, 170, 156)
        self.LINE_COLOR = (23, 145, 135)
        self.CIRCLE_COLOR = (239, 231, 200)
        self.CROSS_COLOR = (66, 66, 66)
        self.TEXT_COLOR = (255, 255, 255)
        
        # Screen setup
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption('Tic Tac Toe')
        self.screen.fill(self.BG_COLOR)
        
        self.draw_lines()
    
    def draw_lines(self):
        """Draw the grid lines"""
        # Horizontal lines
        pygame.draw.line(self.screen, self.LINE_COLOR, 
                        (0, self.SQUARE_SIZE), 
                        (self.WIDTH, self.SQUARE_SIZE), 
                        self.LINE_WIDTH)
        pygame.draw.line(self.screen, self.LINE_COLOR, 
                        (0, 2 * self.SQUARE_SIZE), 
                        (self.WIDTH, 2 * self.SQUARE_SIZE), 
                        self.LINE_WIDTH)
        
        # Vertical lines
        pygame.draw.line(self.screen, self.LINE_COLOR, 
                        (self.SQUARE_SIZE, 0), 
                        (self.SQUARE_SIZE, self.HEIGHT), 
                        self.LINE_WIDTH)
        pygame.draw.line(self.screen, self.LINE_COLOR, 
                        (2 * self.SQUARE_SIZE, 0), 
                        (2 * self.SQUARE_SIZE, self.HEIGHT), 
                        self.LINE_WIDTH)
    
    def draw_figures(self, board):
        """Draw X's and O's based on board state"""
        for row in range(self.BOARD_ROWS):
            for col in range(self.BOARD_COLS):
                if board[row][col] == 2:
                    # Draw O (circle)
                    center = (int(col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2),
                             int(row * self.SQUARE_SIZE + self.SQUARE_SIZE // 2))
                    pygame.draw.circle(self.screen, self.CIRCLE_COLOR, 
                                     center, self.CIRCLE_RADIUS, self.CIRCLE_WIDTH)
                
                elif board[row][col] == 1:
                    # Draw X (cross)
                    start_x = col * self.SQUARE_SIZE + self.SPACE
                    start_y = row * self.SQUARE_SIZE + self.SPACE
                    end_x = col * self.SQUARE_SIZE + self.SQUARE_SIZE - self.SPACE
                    end_y = row * self.SQUARE_SIZE + self.SQUARE_SIZE - self.SPACE
                    
                    pygame.draw.line(self.screen, self.CROSS_COLOR,
                                   (start_x, end_y), (end_x, start_y), 
                                   self.CROSS_WIDTH)
                    pygame.draw.line(self.screen, self.CROSS_COLOR,
                                   (start_x, start_y), (end_x, end_y), 
                                   self.CROSS_WIDTH)
    
    def draw_game_over(self, winner):
        """Draw game over screen with winner or draw message"""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Winner text
        font = pygame.font.Font(None, 74)
        if winner == 1:
            text = font.render('O Wins!', True, self.TEXT_COLOR)
        elif winner == 2:
            text = font.render('X Wins!', True, self.TEXT_COLOR)
        else:
            text = font.render('Draw!', True, self.TEXT_COLOR)
        
        text_rect = text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        self.screen.blit(text, text_rect)
        
        # Restart instructions
        small_font = pygame.font.Font(None, 36)
        restart_text = small_font.render('Press R to Restart', True, self.TEXT_COLOR)
        restart_rect = restart_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 + 50))
        self.screen.blit(restart_text, restart_rect)
        
        quit_text = small_font.render('Press Q to Quit', True, self.TEXT_COLOR)
        quit_rect = quit_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 + 90))
        self.screen.blit(quit_text, quit_rect)
    
    def get_square_from_mouse(self, pos):
        """Convert mouse position to board coordinates"""
        x, y = pos
        row = y // self.SQUARE_SIZE
        col = x // self.SQUARE_SIZE
        return row, col
    
    def clear(self):
        """Clear the screen and redraw grid"""
        self.screen.fill(self.BG_COLOR)
        self.draw_lines()
    
    def update(self):
        """Update the display"""
        pygame.display.update()
    
    def quit(self):
        """Clean up and quit pygame"""
        pygame.quit()
        sys.exit()

class GuiGameController:
    """Controller to manage game state and GUI interactions"""

    def __init__(self, human_symbol, agent, gui, game):
        self.gui = gui
        self.game = game
        self.human_symbol = human_symbol
        self.agent = agent
        self.last_ai_move_time = 0
        self.ai_move_delay = 0.5  # Delay in seconds before AI makes a move

    def reset_game(self):
        """Reset the game state and GUI"""
        self.game.reset()
        self.gui.clear()
        self.last_ai_move_time = 0
        self.gui.update()

    def handle_click(self, pos):
        """Handle mouse click events"""
        if self.game.game_over:
            return

        # Only allow clicks when it's the human's turn
        if self.game.current_player != self.human_symbol:
            return

        row, col = self.gui.get_square_from_mouse(pos)
        if self.game.is_valid_move(row, col):
            self.game.make_move(row, col)
            self.gui.clear()
            self.gui.draw_figures(self.game.board)


    def make_ai_move(self):
        """Make the AI player's move"""
        if self.game.game_over:
            return

        if self.game.current_player != self.agent.symbol:
            return

        # Get move from AI
        move = self.agent.get_move(self.game)
        if move:
            row, col = move
            self.game.make_move(row, col)
            self.gui.clear()
            self.gui.draw_figures(self.game.board)

            if self.game.game_over:
                self.gui.draw_game_over(self.game.winner)

            self.gui.update()
            self.last_ai_move_time = time.time()

    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        self.reset_game()

        while True:
            current_time = time.time()

            # catch pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.gui.quit()
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.game.game_over: # mouse was clicked.
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN: # Keyboard was pressed
                    if event.key == pygame.K_r: # r was clicked
                        self.reset_game()
                    elif event.key == pygame.K_q: # q was clicked
                        self.gui.quit()

            # Make AI move if it's AI's turn
            if not self.game.game_over:
                if self.game.current_player == self.agent.symbol:
                    if current_time - self.last_ai_move_time >= self.ai_move_delay:
                        self.make_ai_move()

            # Rendering
            self.gui.clear()
            self.gui.draw_figures(self.game.board)
            
            if self.game.game_over:
                self.gui.draw_game_over(self.game.winner)
            
            self.gui.update()
            clock.tick(60)


if __name__ == '__main__':
    # Handle commandline arguments
    parser = argparse.ArgumentParser(description='Play Tic Tac Toe against an Agent')
    
    # Argument to choose a symbol for the human player
    parser.add_argument('--player', type=str, choices=['X', 'O'], required=True,
                        help='Choose your symbol: X or O')

    # Argument to choose agent type
    parser.add_argument('--agent', type=str,
                        choices=['minimax', 'qlearn', 'random', 'perfect'],
                        required=True,
                        help='Choose the AI opponent: minimax, qlearn, random, or perfect')

    # Set path to trained q-learn model
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained Q-learning model (required for qlearn agent) ex models/model.pkl')

    args = parser.parse_args()

    # Convert player choice to Symbol
    human_symbol = Symbol.X if args.player == 'X' else Symbol.O
    agent_symbol = Symbol.O if human_symbol == Symbol.X else Symbol.X

    # Create the AI agent based on choice
    agent = None
    if args.agent == 'minimax':
        agent = MinimaxPlayer(agent_symbol)
    elif args.agent == 'qlearn':
        if args.model is None:
            print("Error: --model argument is required when using qlearn agent")
            sys.exit(1)
        agent = QLearnPlayer(agent_symbol, epsilon=0)
        agent.load(args.model)
    elif args.agent == 'random':
        agent = RandomPlayer(agent_symbol)
    elif args.agent == 'perfect':
        agent = PerfectStrategyPlayer(agent_symbol)

    # Create game, GUI, and controller
    game = TicTacToe()
    gui = TicTacToeGUI()
    controller = GuiGameController(human_symbol, agent, gui, game)

    # Run the game
    controller.run()
