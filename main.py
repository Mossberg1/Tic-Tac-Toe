from game.simulator import GameSimulator
from players.minimax_player import MinimaxPlayer
from trainers.qlearn import QLearnTrainer
from players.qlearn_player import QLearnPlayer
from players.random_player import RandomPlayer
from game.symbol import Symbol
import argparse

saved_model = 'models/model.pkl'

parser = argparse.ArgumentParser(description='Tic Tac Toe')
parser.add_argument('-l', '--load', type=str, default=None, help='Path to a existing model to load. If no path is given default model is loaded.')

args = parser.parse_args()

agent_symbol = Symbol.X
opponent_symbol = Symbol.O

agent = QLearnPlayer(agent_symbol)

if args.load is None:
    trainer = QLearnTrainer()

    depths = [1,2,4,6,8,10]
    n_epochs = 100_000

    for depth in depths:
        trainer.train(agent, MinimaxPlayer(opponent_symbol, depth), n_epochs, saved_model)

    trainer.plot()
else:
    agent.load(args.load)

n_simulations = 100

opponent = MinimaxPlayer(opponent_symbol)
simulator = GameSimulator(agent, opponent, n_simulations, Symbol.X)

simulator.simulate()
simulator.plot()


