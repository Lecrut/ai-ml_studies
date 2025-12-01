#%% Imports 
from labyrinthMAP import LabyrinthMap
import random
import math
from tqdm import tqdm
import os
import pickle

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
        self.untried = None

    def is_fully_expanded(self):
        return self.untried is not None and len(self.untried) == 0

    def best_child(self, c=1.414):
        eps = 1e-9

        def uct(ch):
            return (ch.value / (ch.visits + eps)) + c * math.sqrt(
                math.log(self.visits + 1) / (ch.visits + eps)
            )
        return max(self.children, key=uct)

#%% Monte Carlo Tree Search Policy (model-based)
def monte_carlo_tree_search(env, iterations=1000, rollout_limit=100):
    def policy(root_state):
        root = MonteCarloTreeSearchNode(root_state)
        root.untried = list(env.get_only_possible_actions(root_state))

        for _ in range(iterations):
            node = root

            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            if node.untried is None:
                node.untried = list(env.get_only_possible_actions(node.state))

            if node.untried:
                action = node.untried.pop(random.randrange(len(node.untried)))
                next_states = env.get_next_states(node.state, action)
                next_state = next_states[0] if next_states else node.state

                child = MonteCarloTreeSearchNode(next_state, node, action)
                child.untried = list(env.get_only_possible_actions(next_state))
                node.children.append(child)
                node = child

            s = node.state
            total_reward = 0.0

            if node.parent is not None and node.action is not None:
                parent_state = node.parent.state
                total_reward += env.get_reward(parent_state, node.action, node.state)

            for _r in range(rollout_limit):
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
            actions = env.get_only_possible_actions(root_state)
            return random.choice(actions) if actions else None

        best = max(root.children, key=lambda c: c.visits)
        return best.action

    return policy

#%% Precompute policy for ALL states
def precompute_policy(env, iterations=3000, rollout_limit=80):
    print("Precomputing optimal policy for all states...")

    policy_map = {}
    mcts = monte_carlo_tree_search(env, iterations, rollout_limit)

    all_states = env.get_all_states()

    for s in tqdm(all_states, desc="Computing policy"):
        actions = env.get_only_possible_actions(s)
        if not actions:
            continue
        best_action = mcts(s)
        policy_map[s] = best_action

    print("Policy computed!")
    return policy_map

#%% Play game
if __name__ == "__main__":
    env = LabyrinthMap(2)

    policy_file = os.path.join(os.path.dirname(__file__), "records", "policy_MCTS.pkl")
    policy_map = None
    if os.path.exists(policy_file):
        try:
            with open(policy_file, "rb") as f:
                policy_map = pickle.load(f)
            print(f"Loaded policy map from {policy_file}")
        except Exception as e:
            print(f"Failed to load policy map ({e}), will compute a new one.")
            policy_map = None
    else:
        print("No saved policy map found; will compute a new one.")
        
    if policy_map is None:
        policy_map = precompute_policy(env, iterations=2000, rollout_limit=60)
        with open(policy_file, "wb") as f:
            pickle.dump(policy_map, f)
        print(f"Policy map saved to {policy_file}")

    def policy(state):
        if state in policy_map:
            return policy_map[state]
        acts = env.get_only_possible_actions(state)
        return random.choice(acts) if acts else None

    show_game(env, policy, is_show_map=True)
