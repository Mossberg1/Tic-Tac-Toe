from abc import ABC, abstractmethod
from game.symbol import Symbol
from game.logic import TicTacToe


class Player(ABC):
    def __init__(self, symbol: Symbol):
        if symbol != Symbol.X and symbol != Symbol.O:
            raise ValueError()  
        
        self.symbol = symbol
    
    @abstractmethod
    def get_move(self, game: TicTacToe): 
        ... 