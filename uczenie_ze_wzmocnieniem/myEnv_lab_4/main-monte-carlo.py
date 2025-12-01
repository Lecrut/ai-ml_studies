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
                env.show_map()

            if terminal:
                done[i] = True
                print(f"Agent {i} finished with reward: {rewards[i]} in {steps} steps")
                break

    if not any(done):
        print('None agent ended in less than 10000 steps.')  # Corrected message

#%% Monte Carlo Tree Search Node
class MonteCarloTreeSearchNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0
        self.untried = None

    def is_fully_expanded(self):
        return self.untried is not None and len(self.untried) == 0

    def best_child(self, c=1.414):
        return max(self.children,
                   key=lambda ch: ch.value / ch.visits + c * math.sqrt(math.log(self.visits) / ch.visits))

#%% Monte Carlo Tree Search Policy 
def monte_carlo_tree_search(env, iterations=1000, rollout_limit=100, agent_id=0):
    def policy(state):
        root = MonteCarloTreeSearchNode(state)
        root.untried = env.get_possible_actions(state)  # Removed agent_id

        for _ in range(iterations):
            node = root

            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            if node.untried is None:
                node.untried = env.get_possible_actions(node.state) 

            if node.untried:
                action = node.untried.pop()
                next_state, reward, done = env.step(agent_id, action)
                child = MonteCarloTreeSearchNode(next_state, node, action)
                child.untried = env.get_possible_actions(next_state)
                node.children.append(child)
                node = child

            s = node.state
            total_reward = 0
            for _ in range(rollout_limit):
                if env.is_terminal(s):
                    break
                actions = env.get_possible_actions(s)  
                if not actions:
                    break
                a = random.choice(actions)
                s, r, done = env.step(agent_id, a)
                total_reward += r
                if done:
                    break

            while node:
                node.visits += 1
                node.value += total_reward
                node = node.parent

        if not root.children:
            return random.choice(env.get_possible_actions(state))  # Removed agent_id

        return max(root.children, key=lambda c: c.visits).action
    
    return policy

#%% Play game
if __name__ == "__main__":
    env = LabyrinthMap(2)

    states = env.get_all_states()
    # print('States:', states)

    # print_states_and_transitions(env, states)
    policy = monte_carlo_tree_search(env, iterations=1000, rollout_limit=100, agent_id=0)

    show_game(env, policy, is_show_map=True)