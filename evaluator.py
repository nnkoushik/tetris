from game import TetrisGame
from random import randint
#from itertools import accumulate


def accumulate(lis):
    out = []
    acc = 0
    for x in lis:
        acc = acc + x
        out.append(acc)
    return out


def generate_piece():
    pieces = ["S", "Z"]
    return pieces[randint(0,1)]

def monteCarloEval(agent, noOfItems, dataPoints, rows, cols, fileV, fileS):

    stRewards = {}
    states = []
    states.append([[0 for col in range(cols)] for row in range(rows)])
    stRewards[str(states[0])] = [0,0]
    gameBoard = TetrisGame(rows, cols)
    #print(states[0])
    count = 0
    ind_st = 0
    while count < noOfItems:
    #while stRewards[str(states[0])][1] < noOfItems:
        #stInd = randint(0, len(states) - 1)
        agent.reset()
        statesSeen = {}
        rewardSeen = []

        #gameBoard.setNewState(states[0])
        #statesSeen[str(states[0])] = 0
        #print(ind)
        if len(states) < noOfItems:
            gameBoard.setNewState(states[0])
            statesSeen[str(states[0])] = 0
        if len(states) > noOfItems:
            for j in range(ind_st, len(states)):
                temp = str(states[j])
                tup = stRewards[temp]
                if tup[1] < dataPoints:
                    ind_st = j
                    break

            gameBoard.setNewState(states[ind_st])
            statesSeen[str(states[ind_st])] = 0

        while True:
            piece = generate_piece()
            loc = agent.getLocation(piece, gameBoard)
            reward = gameBoard.addPiece(piece, loc[0], loc[1])
            rewardSeen.append(reward)

            if reward < 0:
                break

            strState = str(gameBoard.getState())
            #print(strState)
            if strState not in stRewards:
                states.append(gameBoard.getStateCopy())
            if strState not in statesSeen:
                statesSeen[strState] = len(rewardSeen)
        #print(sum(rewardSeen))
        for visitedSt, ind in statesSeen.items():
            #rewardSum = list(accumulate(list(reversed(rewardSeen))))
            rewardSum = accumulate(list(reversed(rewardSeen)))

            if visitedSt in stRewards:
                tup = stRewards[visitedSt]
                tup[1] = tup[1] + 1
                if tup[1] == dataPoints:
                    count = count + 1
                tup[0] = tup[0] + rewardSum[-(1 + ind)]
                stRewards[visitedSt] = tup
            else:
                stRewards[visitedSt] = [rewardSum[-(1 + ind)], 1]
                if 1 == dataPoints:
                    count = count + 1
        #print(rewardSum[-1])
        #print(count)
        #print(str(len(states)))

    #for key, val in stRewards.items():
    #    print(str(val))
    #    print('\n')
    #print(len(states))
    print(stRewards[str(states[0])])
    #print(sum([sum(x) for x in states[0]]))
    with open(fileV, "w") as V:
        V.write(str(stRewards))
    with open(fileS, "w") as S:
        S.write(str(states))
