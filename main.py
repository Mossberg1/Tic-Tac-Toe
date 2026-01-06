from game.simulator import GameSimulator
from players.random_player import RandomPlayer
from game.symbol import Symbol


player1 = RandomPlayer(Symbol.X)
player2 = RandomPlayer(Symbol.O)

simulator = GameSimulator(player1, player2, 10, Symbol.X)

simulator.simulate()
simulator.plot()