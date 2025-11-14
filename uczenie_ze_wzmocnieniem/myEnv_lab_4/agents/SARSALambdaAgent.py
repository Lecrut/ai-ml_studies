import random
from collections import defaultdict


class SARSALambdaAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions, lambda_value):
        """
        SARSA Lambda Agent
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
        self._evalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.lambda_value = lambda_value

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    def reset(self):
        self._evalues = defaultdict(lambda: defaultdict(lambda: 0))

    # ---------------------START OF YOUR CODE---------------------#

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

        #
        # INSERT CODE HERE to get maximum possible value for a given state
        #


        return max(values)

    def update(self, state, action, reward, next_state):
        """
        You should do your SARSA-Lambda update here:
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha
        next_action = self.get_action(next_state)

        error = reward + gamma * self.get_qvalue(next_state, next_action) - self.get_qvalue(state, action)
        self._evalues[state][action] += 1

        for s in self._qvalues:
            for a in self.get_legal_actions(s):
                old_q = self.get_qvalue(s, a)
                new_q = old_q + learning_rate * error * self._evalues[s][a]
                self.set_qvalue(s, a, new_q)
                self._evalues[s][a] *= (1-learning_rate) * gamma * self.lambda_value
          
        #
        # INSERT CODE HERE to update value in the state for the action 
        #

        return next_action

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
        # INSERT CODE HERE to get best action for a given state
        #
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
        self.epsilon = 0
        self.alpha = 0

    def display_qvalues(self):
        for s in self._qvalues:
            print("State: " + str(s) + " " + str(self._qvalues[s]))

    def __getstate__(self):
        state = self.__dict__.copy()
        state['get_legal_actions'] = None
        state['_qvalues'] = {k: dict(v) for k, v in self._qvalues.items()}
        state['_evalues'] = {k: dict(v) for k, v in self._evalues.items()}
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self._evalues = defaultdict(lambda: defaultdict(lambda: 0))
        for k, v in state['_qvalues'].items():
            self._qvalues[k].update(v)
        for k, v in state['_evalues'].items():
            self._evalues[k].update(v)