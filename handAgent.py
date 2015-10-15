class BasicAgent(object):
    def __init__(self, rows, cols):
        self.Nrows = rows
        self.Ncols = cols
        self.change = True
        self.first = True

    def reset(self):
        self.change = True
        self.first = True
        return 0

    def getLocation(self, piece, gameBoard):
        if self.first:
            self.curr = piece
            self.first = False

        if self.change:
            maxHeight = gameBoard.maxHeight()
            if self.Nrows - maxHeight > 16:
                self.change = False
                self.curr = "Z" if self.curr == "S" else "S"

        if piece == "S":
            h1 = min(gameBoard.columnHeight(0), gameBoard.columnHeight(1))
            h2 = min(gameBoard.columnHeight(2), gameBoard.columnHeight(3))
            pos = 0 if h1 > h2 else 2
            lessOc = h1 if h1 > h2 else h2 
            if (piece == self.curr):
                h3 = min(gameBoard.columnHeight(4), gameBoard.columnHeight(5))
                pos = 4 if h3 > lessOc else pos
                lessOc = h3 if h3 > lessOc else lessOc
        else:
            h1 = min(gameBoard.columnHeight(6), gameBoard.columnHeight(7))
            h2 = min(gameBoard.columnHeight(8), gameBoard.columnHeight(9))
            pos = 6 if h1 > h2 else 8
            lessOc = h1 if h1 > h2 else h2 
            if (piece == self.curr):
                h3 = min(gameBoard.columnHeight(4), gameBoard.columnHeight(5))
                pos = 4 if h3 > lessOc else pos
                lessOc = h3 if h3 > lessOc else lessOc

        return [pos, "V"]