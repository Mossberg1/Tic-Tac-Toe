from player import Player
from game.logic import TicTacToe
from game.symbol import Symbol
from collections import defaultdict
import random
import numpy as np


class QLearnPlayer(Player):    
    """ Q-Learning Agent to play Tic Tac Toe """
    
    def __init__(self, symbol, learning_rate=0.1, discount_rate=0.9, epsilon=0.1):
        super().__init__(self, symbol)
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        
        self._q_table = dict()
    
    
    def get_move(self, game: TicTacToe):
        """ Get the best move """
        state = game.get_board_state()
        moves = game.get_legal_moves()
        
        if len(moves) == 0:
            pass
        
        if random.random() < self.epsilon:
            return random.choice(moves)
        
        action = max(self._q_table[state], key=self._q_table[state].get) # Choose q-value with highest value.
        
        return action
        
    
    def learn(self, last_state, last_action, reward, state):
        """ Make it possible for the agent to learn. """
        row, col = last_action
        action_index = row * 3 + col
        
        try: 
            best_q = np.max(self._q_table[state])
            prev_q = self._q_table[last_state][action_index]
            self._q_table[last_state][action_index] += self._learning_rate * (reward + self._discount_factor * best_q - prev_q)
        except IndexError as ex:
            print(f'Index out of range: {ex}')    
    
    
    def load(self):
        pass
    
    
    def save(self):
        pass
    