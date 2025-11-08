import random

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAP = [
    "ERRRWWWWWE",
    "WWWRRWWWRR",
    "RWWRRWWWRW",
    "RRWRRWRRWR",
    "WRWRRWWRRW",
    "RRRRWWWRRR",
    "WRWWWWWWRW",
    "RRWRRWRRWR",
    "WRWRRWWWRR",
    "ERWWRWWWWE"
]

WIDTH = len(MAP[0])
HEIGHT = len(MAP)

STEP_REWARD = -0.05
GOAL_REWARD = 1.0
COLLISION_PENALTY = -0.5

def get_entrances(num_agents): 
    entrances = []
    for y in [0, HEIGHT - 1]:
        for x in [0, WIDTH - 1]:
            if MAP[y][x] == 'E':
                entrances.append((x, y))

    n_agents = max(2, min(num_agents, len(entrances) - 1))
    return random.sample(entrances, n_agents)

def is_walkable(x, y):
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        return MAP[y][x] != 'W'
    return False

class State: 
    def __init__(self, index, x, y):
        self._index = index
        self._x = x
        self._y = y
    
    def get_position(self):
        return self._x, self._y
    
class Agent:
    def __init__(self, start_state):
        self._start_state = start_state
        self._current_state = start_state

    def get_current_state(self):
        return self._current_state
    
    def set_state(self, state):
        self._current_state = state
    
    def reset(self):
        self._current_state = self._start_state

class LabyrinthMap:
    def __init__(self, num_agents):
        self._num_agents = num_agents
        self._states = []
        for y in range(HEIGHT):
            for x in range(WIDTH):
                if MAP[y][x] != 'W':
                    index = y * WIDTH + x
                    self._states.append(State(index, x, y))
        self._agents = []
        self._goal_state = None
        entrances = get_entrances(num_agents)
        for i, entrance in enumerate(entrances):
            x, y = entrance
            index = y * WIDTH + x
            start_state = next(state for state in self._states if state._index == index)
            if i == len(entrances) - 1:
                self._goal_state = start_state
            else:
                self._agents.append(Agent(start_state))


    def reset(self):
        for agent in self._agents:
            agent.reset()

    def get_all_states(self):
        return self._states

    def is_terminal(self, state):
        return state == self._goal_state

    def get_next_states(self, state, action):
        x, y = state.get_position()
        if action == LEFT:
            x -= 1
        elif action == DOWN:
            y += 1
        elif action == RIGHT:
            x += 1
        elif action == UP:
            y -= 1

        if is_walkable(x, y):
            next_state = next((s for s in self._states if s.get_position() == (x, y)), None)
            return [next_state] if next_state else []
        return []

    def get_possible_actions(self, state):
        return [LEFT, DOWN, RIGHT, UP]

    def get_reward(self, state, action, next_state):
        if next_state == self._goal_state:
            return GOAL_REWARD
        return STEP_REWARD

    def step(self, action):
        rewards = []
        for agent in self._agents:
            current_state = agent.get_current_state()
            next_states = self.get_next_states(current_state, action)
            if next_states:
                next_state = next_states[0]
            else:
                next_state = current_state
            reward = self.get_reward(current_state, action, next_state)
            agent.set_state(next_state)
            rewards.append(reward)

        positions = [agent.get_current_state().get_position() for agent in self._agents]
        if len(positions) != len(set(positions)):
            for agent in self._agents:
                agent.reset()
            rewards = [reward + COLLISION_PENALTY for reward in rewards]

        return rewards
