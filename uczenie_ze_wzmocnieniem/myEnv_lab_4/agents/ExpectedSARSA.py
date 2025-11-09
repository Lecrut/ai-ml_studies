import random
from collections import defaultdict


class ExpectedSARSAAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Q-Learning Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)

        Functions you should use
          - self.get_legal_actions(state) {state, hashable -> list of actions, each is hashable}
            which returns legal actions for a state
          - self.get_qvalue(state,action)
            which returns Q(state,action)
          - self.set_qvalue(state,action,value)
            which sets Q(state,action) := value
        !!!Important!!!
        Note: please avoid using self._qValues directly.
            There's a special self.get_qvalue/set_qvalue for that.
        """

        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    #---------------------START OF YOUR CODE---------------------#

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0
        
        values = []
        for action in possible_actions:
            q_value = self.get_qvalue(state, action)
            values.append(q_value)

        max_value = max(values) if values else 0.0

        #
        # INSERT CODE HERE to get maximum possible value for a given state
        #

        return max_value

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * \sum_a \pi(a|s') Q(s', a))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        current_qvalue = self.get_qvalue(state, action)
        possible_actions = self.get_legal_actions(next_state)

        expected_qvalue = 0.0
        for a in possible_actions:
            q_value = self.get_qvalue(next_state, a)
            prob_a = self.epsilon / len(possible_actions)

            if a == self.get_best_action(next_state):
                prob_a += (1 - self.epsilon)

            expected_qvalue += prob_a * q_value

        updated_qvalue = (1 - learning_rate) * current_qvalue + learning_rate * (reward + gamma * expected_qvalue)
        self.set_qvalue(state, action, updated_qvalue)

        #
        # INSERT CODE HERE to update value for the given state and action
        #


    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None
        
        possible_actions_qvalues = []
        for action in possible_actions:
            q_value = self.get_qvalue(state, action)
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
        """
        Function turns off agent learning.
        """
        self.epsilon = 0
        self.alpha = 0
    