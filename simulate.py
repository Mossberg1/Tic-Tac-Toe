from game.simulator import GameSimulator
from players.random_player import RandomPlayer
from players.minimax_player import MinimaxPlayer
from game.symbol import Symbol


def simulate():
    player1 = RandomPlayer(Symbol.X)
    player2 = MinimaxPlayer(Symbol.O)

    simulator = GameSimulator(player1, player2, 100, Symbol.X)

    simulator.simulate()
    simulator.plot()


if __name__ == "__main__":
    simulate()