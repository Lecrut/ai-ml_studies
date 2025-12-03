#%% Imports 
from labyrinthMAP import LabyrinthMap
import random
import math

#%% Show game
def show_game(env, policy, is_show_map=False):
    env.reset()
    done = [False for _ in range(env._num_agents)]
    rewards = [0 for _ in range(env._num_agents)]
    steps = 0

    while not any(done) and steps < 10000:
        steps += 1
        for i, agent in enumerate(env._agents):
            if done[i]:
                continue

            current_state = agent.get_current_state()
            action = policy(current_state)

            if not is_show_map:
                print(f"Agent {i} at state {current_state} takes action {action}")

            _, reward, terminal = env.step(i, action)
            rewards[i] += reward

            if is_show_map:
                env.show_map(number_of_moves=steps)

            if terminal:
                done[i] = True
                print(f"Agent {i} finished with reward: {rewards[i]} in {steps} steps")
                break

    if not any(done):
        print('None agent ended in less than 10000 steps.')

#%% Monte Carlo Tree Search Node
class MonteCarloTreeSearchNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def best_child(self, c=1.414):
        eps = 1e-9
        def uct(ch):
            return (ch.value / (ch.visits + eps)) + c * math.sqrt(
                math.log(self.visits + 1) / (ch.visits + eps)
            )
        return max(self.children, key=uct)

#%% Online Monte Carlo Tree Search
def mcts_best_action(env, root_state, iterations=800, rollout_limit=60):
    root = MonteCarloTreeSearchNode(root_state)
    root.untried_actions = list(env.get_only_possible_actions(root_state))

    for _ in range(iterations):
        node = root

        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        if node.untried_actions is None:
            node.untried_actions = list(env.get_only_possible_actions(node.state))

        if node.untried_actions:
            action = node.untried_actions.pop(random.randrange(len(node.untried_actions)))
            next_states = env.get_next_states(node.state, action)
            next_state = next_states[0] if next_states else node.state

            child = MonteCarloTreeSearchNode(next_state, node, action)
            child.untried_actions = list(env.get_only_possible_actions(next_state))
            node.children.append(child)
            node = child

        s = node.state
        total_reward = 0.0

        for _ in range(rollout_limit):
            if env.is_terminal(s):
                break

            possible = env.get_only_possible_actions(s)
            if not possible:
                break

            a = random.choice(possible)
            next_states = env.get_next_states(s, a)
            next_s = next_states[0] if next_states else s

            total_reward += env.get_reward(s, a, next_s)
            s = next_s

            if env.is_terminal(s):
                break

        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value += total_reward
            cur = cur.parent

    if not root.children:
        acts = env.get_only_possible_actions(root_state)
        return random.choice(acts) if acts else None

    best = max(root.children, key=lambda c: c.visits)
    return best.action

#%% Play game
if __name__ == "__main__":
    env = LabyrinthMap(2)

    def policy(state):
        return mcts_best_action(env, state, iterations=1200, rollout_limit=80)

    show_game(env, policy, is_show_map=True)
