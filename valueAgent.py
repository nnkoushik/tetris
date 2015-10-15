import TetrisGame from game
import deepcopy from copy

class ValueAgent(object):
    def __init__(self, cols, rows, value):
        self.Ncols = cols
        self.Nrows = rows
        self.value = value

    def reset(self):
        return (self.agent).reset()

    def getLocation(self, piece, gameBoard):
        tempboard = deepcopy(gameBoard)
        max_con = [-1, -1]
        max_val = -100000
        for i in range(0, cols - 1):
            rew = tempboard.addPiece(piece, i, "V")
            rew = rew + self.value.get_value_inp([x for sublist in tempboard.getState() for x in sublist])
            if(rew >= max_val):
                max_val = rew
                max_con = [i, 'V']
            tempboard.setNewState(gameBoard.getState())
        for i in range(0, cols - 2):
            rew = tempboard.addPiece(piece, i, "H")
            rew = rew + self.value.get_value_inp([x for sublist in tempboard.getState() for x in sublist])
            if(rew >= max_val):
                max_val = rew
                max_con = [i, 'H']
            tempboard.setNewState(gameBoard.getState())
        return max_con
