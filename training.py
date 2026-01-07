from trainers.qlearn import QLearnTrainer
from players.qlearn_player import QLearnPlayer
from players.qlearn_player_viktor import QLearnPlayerViktor
from game.symbol import Symbol


def train():
    trainer = QLearnTrainer()
    agent = QLearnPlayerViktor(Symbol.X)
    
    trainer.train(agent, 1000)
    trainer.plot()
    


if __name__ == "__main__":
    train()