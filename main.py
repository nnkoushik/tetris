from evaluator import monteCarloEval
from handAgent import BasicAgent
from epsilonGreedy import EpsilonAgent
from valueAgent import ValueAgent
import pickle
from SdA import SdA

def runEval():
    agent = BasicAgent(20, 10)
    monteCarloEval(agent, 1000, 1, 20, 10, "value.txt", "state.txt")

if __name__ == "__main__":
    runEval()
