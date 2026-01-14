from game.winner_state import WinnerState
from players.player import Player
from players.qlearn_player import QLearnPlayer
from players.minimax_player import MinimaxPlayer
from players.random_player import RandomPlayer
from game.symbol import Symbol
from game.logic import TicTacToe
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# Define rewards and penalty
BASE_REWARD = 0 
WIN_REWARD = 100
DRAW_REWARD = 50
LOSS_PENALTY = -100

# Q-Learning Trainer Class
class QLearnTrainer():
    def __init__(self):
        # Initialize default tracking variables
        self._n_wins = 0
        self._n_draws = 0
        self._n_losses = 0
        self._history = list()
        self._n_games_played = 0
        self._opponent_name = ''
    
    # Train method for Q-Learning agent
    # agent: QLearnPlayer - The Q-Learning agent to be trained
    # opponent: Player - Opponent player, this can be any type of player
    # n_games: int - Number of games to be played during training
    # savepath: str - Path to save the trained model 
    def train(self, agent: QLearnPlayer, opponent: Player, n_games: int, savepath: str):
        self._n_games_played = n_games
        self._opponent_name = type(opponent).__name__
        
        # Create a new game instance of TicTacToe
        game = TicTacToe()
        
        # Loop through the number of games to be played
        for i in tqdm(range(n_games), desc="Training"):
            # Initialize variables to track last state and action
            last_state = None
            last_action = None
            
            # Game loop, run until the game is over
            while game.game_over == False:
                move = None
                
                # Check whose turn it is
                if game.current_player == agent.symbol:
                    # If we are in the agent's turn...

                    # Get the current state of the board                    
                    current_state = game.get_board_state()
                    
                    # If there is a last state, learn from the previous action, give it a base reward
                    if last_state is not None:
                        agent.learn(last_state, last_action, BASE_REWARD, current_state)
                    
                    # Get the agent's move
                    move = agent.get_move(game)
                    
                    # Update last state and action
                    last_state = current_state
                    last_action = move
                    
                    # Make the move on the game board
                    game.make_move(*move)
                    
                else:
                    # If it is the opponent's turn, get their move, based on their strategy
                    move = opponent.get_move(game)
                    # Make the move on the game board
                    game.make_move(*move)
            
            # When the game is done playing, get the current state of the board (the outcome)
            current_state = game.get_board_state()
            
            # Check the winner and give the appropriate reward or penalty
            # and update win/draw/loss counters and history
            match (game.winner):
                case agent.symbol:
                    agent.learn(last_state, last_action, WIN_REWARD, current_state, True)
                    self._n_wins += 1
                    self._history.append(1)
                case WinnerState.DRAW:
                    agent.learn(last_state, last_action, DRAW_REWARD, current_state, True)
                    self._n_draws += 1
                    self._history.append(0)
                case _ :
                    agent.learn(last_state, last_action, LOSS_PENALTY, current_state, True)
                    self._n_losses += 1  
                    self._history.append(-1)
            
            # Reset the game for the next round
            game.reset()
            # Decay epsilon to reduce exploration rate over time
            agent.epsilon *= 0.99995

        # After training is done, save the trained model
        if not savepath.endswith('.pkl'):
            savepath += '.pkl'

        agent.save(savepath)
    
    def plot(self, window_size=500):
        """ Method to plot wins, draws and losses for the tracked agent. """
        if not self._history:
            print("No training history to plot. Please run train() first.")
            return

        df = pd.DataFrame({'outcome': self._history})

        df['Win Rate'] = (df['outcome'] == 1).rolling(window=window_size).mean() * 100
        df['Draw Rate'] = (df['outcome'] == 0).rolling(window=window_size).mean() * 100
        df['Loss Rate'] = (df['outcome'] == -1).rolling(window=window_size).mean() * 100

        plt.figure(figsize=(12, 6))
        
        plt.plot(df.index, df['Win Rate'], label='Win Rate', color='green', linewidth=2)
        plt.plot(df.index, df['Draw Rate'], label='Draw Rate', color='orange', linewidth=1.5, linestyle='--')
        plt.plot(df.index, df['Loss Rate'], label='Loss Rate', color='red', linewidth=1.5, linestyle=':')

        plt.title(f'Learning Curve vs {self._opponent_name}')
        plt.xlabel('Game Number')
        plt.ylabel('Rate (%)')
        plt.legend(loc='center right')
        plt.grid(True, alpha=0.3)
        plt.ylim(-5, 105)

        plt.show()

        
            
            




