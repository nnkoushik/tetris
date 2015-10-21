from game import TetrisGame
from copy import deepcopy

class ValueAgent(object):
    def __init__(self, cols, rows, value):
        self.Ncols = cols
        self.Nrows = rows
        self.value = value

    def reset(self):
#        return (self.agent).reset()
        return 0
    def getLocation(self, piece, gameBoard):
        tempboard = deepcopy(gameBoard)
        max_con = [0, 'V']
        max_val = -float("infinity")
        for i in range(0, self.Ncols - 1):
            rew = tempboard.addPiece(piece, i, "V")
            rew = rew + self.value.compute_val([x for sublist in tempboard.getState() for x in sublist]).eval()
            if(rew >= max_val):
                max_val = rew
                max_con = [i, 'V']
            tempboard.setNewState(gameBoard.getState())
        for i in range(0, self.Ncols - 2):
            rew = tempboard.addPiece(piece, i, "H")
            rew = rew + self.value.compute_val([x for sublist in tempboard.getState() for x in sublist]).eval()
            if(rew >= max_val):
                max_val = rew
                max_con = [i, 'H']
            tempboard.setNewState(gameBoard.getState())
        return max_con
