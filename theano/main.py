from evaluator import monteCarloEval
from handAgent import BasicAgent
from epsilonGreedy import EpsilonAgent

def runEval():
    agent = EpsilonAgent(10, BasicAgent(20, 10), 0.1)
    monteCarloEval(agent, 50000, 1, 20, 10, "value.txt", "state.txt")

if __name__ == "__main__":
    runEval()
