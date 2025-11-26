import random
import os
import time
import itertools

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAP = [
    "REEERWRRRRR",
    "RRRRRRRWRWW",
    "RWRWRRWRRRR",
    "WRRRWRWWRWR",
    "RRWWRRRRRWR",
    "WRWRRWRWWRW",
    "WRRRRWRRWRR",
    "RRWRRWWRRRW",
    "RWRWRWRRRWR",
    "RWRRRRRWRRG"
]
# MAP = ['EEE', 'RWR', 'RGR']

WIDTH = len(MAP[0])
HEIGHT = len(MAP)

STEP_REWARD = -0.5
GOAL_REWARD = 100.0
COLLISION_PENALTY = -5.0

def get_entrances(num_agents): 
    entrances = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if MAP[y][x] == 'E':
                entrances.append((x, y))

    n_agents = max(1, min(num_agents, len(entrances)))
    return random.sample(entrances, n_agents)

def get_goal_position():
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if MAP[y][x] == 'G':
                return (x, y)
    return None

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
    
    def __hash__(self):
        return hash((self._x, self._y))
    
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return self._x == other._x and self._y == other._y
    
    def __repr__(self):
        return f"State({self._x}, {self._y})"
    
# ...existing code...
class Agent:
    def __init__(self, start_state, index, positions=()):
        self._start_state = start_state
        self._current_state = start_state
        self._index = index
        self._start_positions = tuple(positions) if positions is not None else ()
        self._positions = tuple(positions) if positions is not None else ()

    def get_current_state(self):
        results = [self._current_state] + list(self._positions)
        state_results = []
        for res in results: 
            if isinstance(res, State):
                state_results.append(res)
                continue
            x, y = res
            index = y * WIDTH + x
            state_results.append(State(index, x, y))

        return tuple(state_results)
    
    def get_current_position(self):
        return self._current_state
    
    def set_state(self, state, positions):
        self._positions = tuple(positions) if positions is not None else ()
        self._current_state = state[0]
    
    def reset(self):
        self._current_state = self._start_state
        self._positions = self._start_positions

    def __hash__(self):
        return hash((self._current_state, self._positions))

    def __eq__(self, other):
        if not isinstance(other, Agent):
            return False
        return (self._current_state == other._current_state and
                self._positions == other._positions and
                self._index == other._index)

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

        x_goal, y_goal = get_goal_position()
        index_goal = y_goal * WIDTH + x_goal
        self._goal_state = next(state for state in self._states if state._index == index_goal)

        entrances = get_entrances(num_agents)
        for i, entrance in enumerate(entrances):
            x, y = entrance
            index = y * WIDTH + x
            start_state = next(state for state in self._states if state._index == index)
            other_positions = [e for j, e in enumerate(entrances) if j != i]
            self._agents.append(Agent(start_state, i, other_positions))

    def reset(self):
        for agent in self._agents:
            agent.reset()

    def get_all_states(self):
        available_states = list(itertools.product(self._states, repeat=self._num_agents))
        return available_states
    
    def get_next_move(self, agent_idx, policy):
        agent = self._agents[agent_idx]
        current_state = agent.get_current_state()
        action = policy[current_state]
        return action

    def get_next_states(self, state, action):
        position = state[0]
        x, y = position.get_position()
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
            return [(next_state,) + state[1:]]
        return [state] 
    
    def get_only_possible_actions(self, state):
        agent_pos = state[0]
        x, y = agent_pos.get_position()
        actions = []

        if is_walkable(x-1, y):
            actions.append(LEFT)
        if is_walkable(x+1, y):
            actions.append(RIGHT)
        if is_walkable(x, y-1):
            actions.append(UP)
        if is_walkable(x, y+1):
            actions.append(DOWN)

        if not actions:
            actions.append(None) 

        return actions
        
    def get_possible_actions(self, state):
        return [LEFT, DOWN, RIGHT, UP]

    def is_terminal(self, state):
        if isinstance(state, (tuple, list)):
            return state[0] == self._goal_state
        return state == self._goal_state

    def get_reward(self, state, action, next_state):
        ns = next_state[0] if isinstance(next_state, (tuple, list)) else next_state
        if ns == self._goal_state:
            return GOAL_REWARD
        return STEP_REWARD

    def step(self, agent_idx, action):
        agent = self._agents[agent_idx]
        cur = agent.get_current_state()

        candidates = self.get_next_states(cur, action)
        next_state = candidates[0] if candidates else cur[0] 

        reward = self.get_reward(cur, action, next_state)

        for updated_agent in self._agents:
            agents_positions = []
            for other_agent in self._agents:
                if other_agent == updated_agent:
                    continue
                agents_positions.append(other_agent.get_current_position().get_position())
            if agent == updated_agent:
                agent.set_state(next_state, agents_positions)
            else:
                updated_agent.set_state(updated_agent.get_current_state(), agents_positions)

        if self.is_terminal(next_state):
            return next_state, GOAL_REWARD, True

        positions = [ag.get_current_position().get_position() for ag in self._agents]
        if len(positions) != len(set(positions)):
            collided_positions = {p for p in positions if positions.count(p) > 1}
            for ag in self._agents:
                if ag.get_current_position().get_position() in collided_positions:
                    ag.reset()
            reward += COLLISION_PENALTY

        return agent.get_current_state(), reward, False

    def show_map(self, clear=True, delay=0.1):
        if clear:
            os.system('cls')

        RESET = "\x1b[0m"
        GREEN = "\x1b[32;1m"
        RED = "\x1b[31;1m"
        WHITE = "\x1b[37;1m"

        base = []
        for y in range(HEIGHT):
            row = []
            for x in range(WIDTH):
                ch = MAP[y][x]
                row.append('#' if ch == 'W' else ' ')
            base.append(row)

        padded = []
        border_row = ['#'] * (WIDTH + 2)
        padded.append(border_row[:])
        for y in range(HEIGHT):
            padded.append(['#'] + base[y] + ['#'])
        padded.append(border_row[:])

        for idx, agent in enumerate(self._agents, start=1):
            ax, ay = agent.get_current_position().get_position()
            px, py = ax + 1, ay + 1
            padded[py][px] = str(idx)

        gx, gy = self._goal_state.get_position()
        padded[gy + 1][gx + 1] = 'G'

        for row in padded:
            line_parts = []
            for ch in row:
                if ch == '#':
                    line_parts.append(f"{WHITE}#{RESET}")
                elif ch == 'G':
                    line_parts.append(f"{RED}G{RESET}")
                elif ch.isdigit():
                    line_parts.append(f"{GREEN}{ch}{RESET}")
                else:
                    line_parts.append(' ')
            print(''.join(line_parts), flush=True)

        if delay and delay > 0:
            time.sleep(delay)