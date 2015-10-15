from copy import deepcopy

class TetrisGame(object):
    def __init__(self, rows, cols):
        self.Nrows = rows
        self.Ncols = cols
        self.currState = 0

    def getState(self):
        return self.currState

    def setNewState(self, newState):
        self.currState = deepcopy(newState)

    def getStateCopy(self):
        return deepcopy(self.currState)

    def columnHeight(self, column):
        for height in range(0, self.Nrows):
            #print(height)
            if self.currState[height][column] == 1:
                return height
        return self.Nrows

    def maxHeight(self):
        flag = 0
        for i in range(0,self.Nrows):
            for j in range(0, self.Ncols):
                if(self.currState[i][j] == 1):
                    flag = 1
                    break
            if flag == 1:
                return i
        return self.Nrows

    def fullRow(self, rowNo):
        for j in range(0, self.Ncols):
            if(self.currState[rowNo][j] == 0):
                return False

        return True

    def clearRow(self, rowNo):
        if (self.fullRow(rowNo)):
            for j in range(0, self.Ncols):
                for i in reversed(range(1, rowNo + 1)):
                    self.currState[i][j] = self.currState[i - 1][j]
                self.currState[0][j] = 0

            return 1
            
        return 0;

    def addPiece(self, piece, pos, rot):
        reward = 0
        if (piece == "S" and rot == "V"):
            h1 = self.columnHeight(pos)
            h2 = self.columnHeight(pos + 1)

            if(h2 <= h1):
                if(h2 < 3): return -1
                self.currState[h2 - 3][pos] = 1
                self.currState[h2 - 2][pos] = 1
                self.currState[h2 - 2][pos + 1] = 1
                self.currState[h2 - 1][pos + 1] = 1
                
                reward = self.clearRow(h2 - 1)
                reward = reward + self.clearRow(reward + h2 - 2)
            else:
                if(h1 < 2): return -1
                self.currState[h1 - 2][pos] = 1
                self.currState[h1 - 1][pos] = 1
                self.currState[h1 - 1][pos + 1] = 1
                self.currState[h1][pos + 1] = 1
                    
                reward = self.clearRow(h1)
                reward = reward + self.clearRow(reward + h1 - 1)
        if (piece == "Z" and rot == "V"):
            h1 = self.columnHeight(pos)
            h2 = self.columnHeight(pos + 1)

            if(h1 <= h2):
                if(h1 < 3): return -1
                self.currState[h1 - 3][pos + 1] = 1
                self.currState[h1 - 2][pos + 1] = 1
                self.currState[h1 - 2][pos] = 1
                self.currState[h1 - 1][pos] = 1
                    
                reward = self.clearRow(h1 - 1)
                reward = reward + self.clearRow(reward + h1 - 2)
            else:
                if(h2 < 2): return -1
                self.currState[h2 - 2][pos + 1] = 1
                self.currState[h2 - 1][pos + 1] = 1
                self.currState[h2 - 1][pos] = 1
                self.currState[h2][pos] = 1
                    
                reward = self.clearRow(h2)
                reward = reward + self.clearRow(reward + h2 - 1)
        if (piece == "S" and rot == "H"):
            h1 = self.columnHeight(pos)
            h2 = self.columnHeight(pos + 1)
            h3 = self.columnHeight(pos + 2)

            if (h1 <= h3 or h2 <= h3):
                minH = h1 if (h1 < h2) else h2
                    
                if(minH < 2): return -1
                    
                self.currState[minH - 1][pos] = 1
                self.currState[minH - 1][pos + 1] = 1
                self.currState[minH - 2][pos + 1] = 1
                self.currState[minH - 2][pos + 2] = 1
                    
                reward = self.clearRow(minH - 1)
            else:
                if(h3 < 1): return -1    

                self.currState[h3][pos] = 1
                self.currState[h3][pos + 1] = 1
                self.currState[h3 - 1][pos + 1] = 1
                self.currState[h3 - 1][pos + 2] = 1

                reward = self.clearRow(h3)
        if (piece == "Z" and rot == "H"):
            h1 = self.columnHeight(pos)
            h2 = self.columnHeight(pos + 1)
            h3 = self.columnHeight(pos + 2)

            if (h2 <= h1 or h3 <= h1):
                minH = h2 if (h2 < h3) else h3
                    
                if(minH < 2): return -1
                    
                self.currState[minH - 2][pos] = 1
                self.currState[minH - 2][pos + 1] = 1
                self.currState[minH - 1][pos + 1] = 1
                self.currState[minH - 1][pos + 2] = 1
                    
                reward = self.clearRow(minH - 1)
            else:
                if(h1 < 1): return -1

                self.currState[h1 - 1][pos] = 1
                self.currState[h1 - 1][pos + 1] = 1
                self.currState[h1][pos + 1] = 1
                self.currState[h1][pos + 2] = 1
                    
                reward = self.clearRow(h1)
        return reward
