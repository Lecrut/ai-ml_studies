#%% Imports 
import random
import math
import copy
from labyrinthMAP import LabyrinthMap, GOAL_REWARD, STEP_REWARD

#%% Manhattan distance to goal
def dist_to_goal(state, env):
    x, y = state[0].get_position()
    gx, gy = env._goal_state.get_position()
    return abs(x - gx) + abs(y - gy)

#%% Check if game already won
def game_already_won(env):
    return any(env.is_terminal(ag.get_current_state()) for ag in env._agents)

#%% Smart Rollout - simulate until terminal or max steps
def smart_rollout(env, start_state, player_id):
    sim = copy.deepcopy(env)
    sim._agents[player_id]._current_state = start_state[0]

    if game_already_won(sim):
        return GOAL_REWARD / 100.0

    steps = 0
    while steps < 40:
        steps += 1

        state = sim._agents[player_id].get_current_state()
        if sim.is_terminal(state):
            return GOAL_REWARD / 100.0

        actions = [a for a in sim.get_only_possible_actions(state) if a is not None]
        if not actions:
            break
        
        if random.random() < 0.95:
            action = min(actions, key=lambda a: dist_to_goal(sim.get_next_states(state, a)[0], sim))
        else:
            action = random.choice(actions)

        _, _, terminal = sim.step(player_id, action)

        if terminal:
            return GOAL_REWARD / 100.0

    return max(0.0, GOAL_REWARD + STEP_REWARD * (steps + dist_to_goal(sim._agents[player_id].get_current_state(), sim))) / 100.0

#%% Monte Carlo Tree Search Agent
class MonteCarloNode:
    def __init__(self, state, parent=None, action=None, player=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.player = player
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried = None
    
    def score(self, ch, c):
            if ch.visits == 0: 
                return float('inf')
            
            return ch.value / ch.visits + c * math.sqrt(math.log(self.visits + 1) / (ch.visits + 1))

    def best_child(self, c=0.4):
        return max(self.children, key=lambda ch: self.score(ch, c))

#%% MCTS Best Action
def mcts_best_action(env, root_state, player, iterations=600):
    root = MonteCarloNode(root_state, player=player)
    root.untried = [a for a in env.get_only_possible_actions(root_state) if a is not None]

    for _ in range(iterations):
        node = root
        while not node.untried and node.children:
            node = node.best_child(0.4)

        if node.untried:
            next_action = random.choice(node.untried)
            node.untried.remove(next_action)
            next_state = env.get_next_states(node.state, next_action)[0]
            child = MonteCarloNode(next_state, node, next_action, player)
            child.untried = [a for a in env.get_only_possible_actions(next_state) if a is not None]
            node.children.append(child)
            node = child

        value = smart_rollout(env, node.state, player)
        while node:
            node.visits += 1
            node.value += value
            node = node.parent

    if not root.children:
        valid_actions = [a for a in env.get_only_possible_actions(root_state) if a is not None]

        if valid_actions:
            return random.choice(valid_actions)

        return None

    return max(root.children, key=lambda c: c.value / c.visits if c.visits else 0).action

#%% Policy function
def policy(state, player_id):
    return mcts_best_action(env, state, player_id, iterations=600)

#%% Show game
def show_game(env, policy, show_map=True):
    env.reset()
    steps = 0
    done = [False for _ in range(env._num_agents)]
    rewards = [0 for _ in range(env._num_agents)]

    while not any(done) and steps < 1000:
        steps += 1

        for i in range(env._num_agents):
            action = policy(env._agents[i].get_current_state(), i)
            _, reward, terminal = env.step(i, action)
            rewards[i] += reward

            if show_map:
                env.show_map(number_of_moves=steps)
            
            if terminal:
                done[i] = True
                print(f"Agent {i} finished with reward: {rewards[i]} in {steps} steps")
                break

    if not any(done):
        print('None agent ended in less than 200 steps.')    

#%% Play game
if __name__ == "__main__":
    env = LabyrinthMap(2)   

    show_game(env, policy, show_map=True)