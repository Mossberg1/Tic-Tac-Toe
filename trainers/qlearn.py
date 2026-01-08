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


BASE_REWARD = 0 
WIN_REWARD = 100
DRAW_REWARD = 50
LOSS_PENALTY = -100


class QLearnTrainer():
    def __init__(self):
        self._n_wins = 0
        self._n_draws = 0
        self._n_losses = 0
        self._history = list()
        self._n_games_played = 0
        self._opponent_name = ''
    
    
    def train(self, agent: QLearnPlayer, opponent: Player, n_games: int, savepath: str):
        self._n_games_played = n_games
        self._opponent_name = type(opponent).__name__
        
        game = TicTacToe()
        
        for i in tqdm(range(n_games), desc="Training"):
            last_state = None
            last_action = None
            
            while game.game_over == False:
                move = None
                
                if game.current_player == agent.symbol:                    
                    current_state = game.get_board_state()
                    
                    if last_state is not None:
                        agent.learn(last_state, last_action, BASE_REWARD, current_state)
                    
                    move = agent.get_move(game)
                    
                    last_state = current_state
                    last_action = move
                    
                    game.make_move(*move)
                    
                else:
                    move = opponent.get_move(game)
                    game.make_move(*move)
            
            current_state = game.get_board_state()
            
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
            
            game.reset()
            agent.epsilon *= 0.99995

        if not savepath.endswith('.pkl'):
            savepath += '.pkl'

        agent.save(savepath)
    
    
    def plot(self):
        """ Method to plot wins, draws and losses for the tracked agent. """
        df = pd.DataFrame(self._history, columns=['result'])
        print(self._n_games_played)
        window_size = int(self._n_games_played / 10)
        df['win_rate'] = df['result'].apply(lambda x: 1 if x == 1 else 0).rolling(window=window_size).mean()
        df['loss_rate'] = df['result'].apply(lambda x: 1 if x == -1 else 0).rolling(window=window_size).mean()
        df['draw_rate'] = df['result'].apply(lambda x: 1 if x == 0 else 0).rolling(window=window_size).mean()

        plt.figure(figsize=(12, 6))
        plt.plot(df['win_rate'], label='Win Rate', color='green')
        plt.plot(df['draw_rate'], label='Draw Rate', color='blue', linestyle='--')
        plt.plot(df['loss_rate'], label='Loss Rate', color='red', linestyle='--')
        
        plt.title(f'Agent Learning Curve against: {self._opponent_name}')
        plt.xlabel('Games Played')
        plt.ylabel('Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


        
            
            




