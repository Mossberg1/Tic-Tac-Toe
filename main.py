# Import Modules
from game.simulator import GameSimulator
from players.minimax_player import MinimaxPlayer
from trainers.qlearn import QLearnTrainer
from players.qlearn_player import QLearnPlayer
from players.random_player import RandomPlayer
from players.perfect_strategy_player import PerfectStrategyPlayer
from game.symbol import Symbol
import argparse

# Store model path
saved_model = 'models/model.pkl'

# Parse arguments from command line
parser = argparse.ArgumentParser(description='Tic Tac Toe')
parser.add_argument('-l', '--load', type=str, default=None, help='Path to a existing model to load. If no path is given default model is loaded.')

args = parser.parse_args()

# Assign symbols to each agent
agent_symbol = Symbol.X
opponent_symbol = Symbol.O

# If no model is provided, we train a new model
if args.load is None:
    trainer = QLearnTrainer()
    agent = QLearnPlayer(agent_symbol)

    # Define depths and epochs for training
    depths = [1,2,4,6,8,10]
    n_epochs = 100_000

    # Train against a random player first
    trainer.train(agent, RandomPlayer(opponent_symbol), n_epochs, saved_model)

    # Loop through each depth and train the agent n_epochs of times
    for depth in depths:
        trainer.train(agent, MinimaxPlayer(opponent_symbol, depth), n_epochs, saved_model)

    trainer.plot()

    #Second training with reversed symbols
    # agent_symbol = Symbol.O
    # opponent_symbol = Symbol.X
    # agent = QLearnPlayer(agent_symbol)
    # agent.load(saved_model)
    #
    # depths = [1,2,4,6,8,10]
    # n_epochs = 100_000
    #
    # trainer.train(agent, RandomPlayer(opponent_symbol), n_epochs, saved_model)
    #
    # for depth in depths:
    #     trainer.train(agent, MinimaxPlayer(opponent_symbol, depth), n_epochs, saved_model)
    #
    # trainer.plot()


else:
    # If there is an existing model, load it
    agent = QLearnPlayer(agent_symbol, epsilon=0)
    agent.load(args.load)

# Define number of simulations
n_simulations = 100

# Simulate against a random player
opponent = RandomPlayer(opponent_symbol)
simulator = GameSimulator(agent, opponent, n_simulations, Symbol.X)
simulator.simulate()
simulator.plot()

# Simulate against a perfect strategy player
opponent = PerfectStrategyPlayer(opponent_symbol)
simulator = GameSimulator(agent, opponent, n_simulations, Symbol.X)
simulator.simulate()
simulator.plot()

# Simulate against a minimax player
opponent = MinimaxPlayer(opponent_symbol)
simulator = GameSimulator(agent, opponent, n_simulations, Symbol.X)
simulator.simulate()
simulator.plot()

