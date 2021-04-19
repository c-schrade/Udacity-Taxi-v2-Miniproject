import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, epsilon, alpha, gamma=1, nA=6, nS=500):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.nS = nS
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.policy = np.zeros([nS,nA])
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        a_max = np.argmax(self.Q[state][:])

        probs = [(1-self.epsilon+self.epsilon/self.nA if a==a_max
                  else self.epsilon/self.nA) for a in range(self.nA)]

        self.policy[state][:] = probs

        action = np.random.choice(np.arange(self.nA),p=probs)

        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        self.Q[state][action] += self.alpha*(reward+self.gamma*np.sum(
                                  self.policy[next_state][:]*self.Q[next_state][:])-self.Q[state][action])
