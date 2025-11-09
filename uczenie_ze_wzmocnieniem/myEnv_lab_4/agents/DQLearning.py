import random
from collections import defaultdict


class DQLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Double Q-Learning Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)
        """

        self.get_legal_actions = get_legal_actions
        self._qvaluesA = defaultdict(lambda: defaultdict(lambda: 0))
        self._qvaluesB = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        
    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvaluesA[state][action] + self._qvaluesB[state][action] 


    #---------------------START OF YOUR CODE---------------------#

    def get_value_chose(self, state, action, isA = True):
        if isA:
            return self._qvaluesA[state][action]
        return self._qvaluesB[state][action]

    def get_best_action(self, state, isA = True):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None
        
        possible_actions_qvalues = []
        for action in possible_actions:
            q_value = self.get_value_chose(state, action, isA)
            possible_actions_qvalues.append(q_value)
        
        max_qvalue = max(possible_actions_qvalues)
        best_actions = []
        for i in range(len(possible_actions)):
            a = possible_actions[i]
            q = possible_actions_qvalues[i]
            if q == max_qvalue:
                best_actions.append(a)

        best_action = random.choice(best_actions)

        #
        # INSERT CODE HERE to get best possible action in a given state (remember to break ties randomly)
        #

        return best_action

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        prob = random.uniform(0, 1)
        if prob < 0.5:
            self._qvaluesA[state][action] += learning_rate * (reward + gamma * self._qvaluesB[next_state][self.get_best_action(next_state, isA=True)] - self._qvaluesA[state][action])
        else:
            self._qvaluesB[state][action] += learning_rate * (reward + gamma * self._qvaluesA[next_state][self.get_best_action(next_state, isA=False)] - self._qvaluesB[state][action])

        #
        # INSERT CODE HERE to update value in the state for the action 
        #


    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        prob = random.uniform(0, 1)
        if prob < epsilon:
            chosen_action = random.choice(possible_actions)
        else:
            chosen_action = self.get_best_action(state)

        #
        # INSERT CODE HERE to get action in a given state (according to epsilon greedy algorithm)
        #        

        return chosen_action

    def turn_off_learning(self):
        self.epsilon = 0
        self.alpha = 0
