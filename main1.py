from evaluator import monteCarloEval
from handAgent import BasicAgent
from epsilonGreedy import EpsilonAgent
from valueAgent import ValueAgent
import pickle
from SdA import SdA

def runEval():
    f = pickle.load(open('gen.pickle', 'rb'))
    agent = ValueAgent(10, 20, f)
    monteCarloEval(agent, 1000, 1, 20, 10, "value1.txt", "state1.txt")

if __name__ == "__main__":
    runEval()
