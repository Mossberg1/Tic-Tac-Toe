from players.player import Player
from typing import Optional
from game.logic import TicTacToe
from game.symbol import Symbol
from collections import defaultdict
import random
import pickle


def default_value():
    return defaultdict(float)


class QLearnPlayer(Player):    
    """ Q-Learning Agent to play Tic Tac Toe """
    
    def __init__(self, symbol, learning_rate=0.1, discount_rate=0.9, epsilon=0.1):
        super().__init__(symbol)
        self.learning_rate = learning_rate  # α (alpha)
        self.discount_rate = discount_rate  # γ (gamma)
        self.epsilon = epsilon  # exploration rate
        
        self._q_table = defaultdict(default_value)
    
    
    def get_move(self, game: TicTacToe) -> Optional[tuple[int, int]]:
        """ Get the best move using epsilon-greedy policy """
        state = game.get_board_state()
        moves = game.get_legal_moves()
        
        if len(moves) == 0:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(moves)
        
        #try:
        q_values = {move: self._q_table[state][move] for move in moves}
        #except KeyError:
            #return random.choice(moves)
        
        best_action = max(q_values, key=q_values.get)
        return best_action
    
    
    def learn(self, last_state, last_action, reward, next_state, done=False):

        current_q = self._q_table[last_state][last_action]
        
        if done:
            self._q_table[last_state][last_action] = reward
            return

        next_state_q_values = self._q_table[next_state].values()
        max_future_q = max(next_state_q_values) if next_state_q_values else 0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_rate * max_future_q - current_q)
        
        self._q_table[last_state][last_action] = new_q
    
    def load(self, filename):
        """ Load Q-table from file """
        with open(filename, 'rb') as f:
            self._q_table = pickle.load(f)
    
    
    def save(self, filename):
        """ Save Q-table to file """
        with open(filename, 'wb+') as f:
            pickle.dump(self._q_table, f)

