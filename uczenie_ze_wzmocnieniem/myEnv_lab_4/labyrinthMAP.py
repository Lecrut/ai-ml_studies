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
    "WWWRRWWWRR",
    "EWWWRWWWWE"
]

WIDTH = len(MAP[0])
HEIGHT = len(MAP)

class LabyrinthMap:
    def __init__(self):
        self._states = [i for i in range(WIDTH * HEIGHT) if MAP[i // WIDTH][i % WIDTH] != 'W']
        pass

    def reset(self):
        pass

    def get_all_states(self):
        pass

    def is_terminal(self, state):
        pass

    def get_possible_actions(self, state):
        pass

    def get_next_states(self, state, action):
        pass

    def get_possible_actions(self, state):
        pass

    def get_reward(self, state, action, next_state):
        pass

    def step(self, action):
        pass
