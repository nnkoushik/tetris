from random import random
from random import randint

class EpsilonAgent(object):
    def __init__(self, cols, agent, epsilon):
        self.Ncols = cols
        self.epsilon = epsilon
        self.agent = agent

    def reset(self):
        return (self.agent).reset()

    def getLocation(self, piece, gameBoard):
        if random() > self.epsilon:
            return (self.agent).getLocation(piece, gameBoard)
        else:
            orientation = "V" if randint(0,1) == 1 else "H"
            if orientation == "V":
                pos = randint(0, self.Ncols - 2)
            else:
                pos = randint(0, self.Ncols - 3)

            return [pos, orientation]
