#%% Imports
from agents.QLearningAgent import QLearningAgent
from agents.DQLearningAgent import DQLearningAgent
from agents.SARSALambdaAgent import SARSALambdaAgent
from agents.SARSAAgent import SARSAAgent
from agents.ExpectedSARSAAgent import ExpectedSARSAAgent
import random
from labyrinthMAP import LabyrinthMap
from tqdm import trange
import matplotlib.pyplot as plt
import pickle
import os

#%% Save and Load Functions
def load_agent(agent_name, legal_actions=None):
    try: 
        with open(f"records/{agent_name}.pkl", "rb") as f:
            agent, score = pickle.load(f)
        if legal_actions is not None:
            agent.get_legal_actions = legal_actions
        return agent, score
    except FileNotFoundError:
        return None, float('-inf')
    except Exception:
        return None, float('-inf')


def save_agent(agent, agent_name, score):
    os.makedirs("records", exist_ok=True)
    _, best_score = load_agent(agent_name)

    if score <= best_score:
        return
    try:
        with open(f"records/{agent_name}.pkl", "wb") as f:
            pickle.dump((agent, score), f)
    except Exception as e:
        print(f"Error saving agent {agent_name}: {e}")
        return

#%% Create agent
def create_agent(agent_type, alpha, epsilon, discount, legal_actions):
    match agent_type:
        case 'QLearning':
            return QLearningAgent(alpha=alpha, epsilon=epsilon, discount=discount,
                                  get_legal_actions=legal_actions)
        case 'DQLearning':
            return DQLearningAgent(alpha=alpha, epsilon=epsilon, discount=discount,
                                   get_legal_actions=legal_actions)
        case 'SARSALambda':
            return SARSALambdaAgent(alpha=alpha, epsilon=epsilon, discount=discount,
                                    get_legal_actions=legal_actions, lambda_value=0.9)
        case 'SARSA':
            return SARSAAgent(alpha=alpha, epsilon=epsilon, discount=discount,
                              get_legal_actions=legal_actions)
        case 'ExpectedSARSA':
            return ExpectedSARSAAgent(alpha=alpha, epsilon=epsilon, discount=discount,
                                      get_legal_actions=legal_actions)
        case _:
            return QLearningAgent(alpha=alpha, epsilon=epsilon, discount=discount,
                                  get_legal_actions=legal_actions)

#%% Training Function
def training(agents_name, is_show_plots=False):
    num_agents = len(agents_name)
    env = LabyrinthMap(num_agents)

    max_tests = 400
    n_eps = 300
    eps = 0.9
    lr = 0.1
    discount = 0.9
    best_agents_rewards = []

    for _ in trange(max_tests):
        eps -= 0.002
        # agents = [create_agent(name, lr, eps, discount, env.get_possible_actions) for name in agents_name]
        agents = []
        for name in agents_name:
            loaded_agent, _ = load_agent(name, env.get_possible_actions)
            if loaded_agent:
                agents.append(loaded_agent)
            else:
                agents.append(create_agent(name, lr, eps, discount, env.get_possible_actions))

        rewards = [0 for _ in range(num_agents)]
        for _ in range(n_eps):
            env.reset()
            done = [False for _ in range(num_agents)]
            while not any(done):
                for i, agent in enumerate(agents):
                    if done[i]:
                        continue
                    state = env._agents[i].get_current_state()
                    action = agent.get_action(state)
                    next_state, reward, terminal = env.step(i, action)
                    agent.update(state, action, reward, next_state)
                    rewards[i] += reward
                    if terminal:
                        done[i] = True
                        best_agents_rewards.append(rewards[i])
                        save_agent(agent, agents_name[i], rewards[i])
                        break

    print(f"Training completed. Rewards: {max(best_agents_rewards)}")

    if is_show_plots:
        plt.plot(best_agents_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('Rewards over Episodes during Training')
        plt.show()

#%% Show Game 
def show_game(agents_name):
    print(f"Showing game for agents: {agents_name}")
    num_agents = len(agents_name)

    env = LabyrinthMap(num_agents)
    env.reset()
    done = False
    # agents = [create_agent(name, alpha=0, epsilon=0, discount=1, legal_actions=env.get_possible_actions) for name in agents_name]
    agents = []
    for name in agents_name:
        loaded_agent, _ = load_agent(name, env.get_possible_actions)
        if loaded_agent:
            agents.append(loaded_agent)
        else:
            agents.append(create_agent(name, alpha=0, epsilon=0, discount=1, legal_actions=env.get_possible_actions))

    for agent in agents:
        agent.turn_off_learning()

    while not done:
        for i, agent in enumerate(agents):
            state = env._agents[i].get_current_state()
            action = agent.get_action(state)
            _, _, terminal = env.step(i, action)

            env.show_map()
            if terminal:
                done = True
                break

#%% Run Game
if __name__ == "__main__":
    available_agents = ['QLearning', 'DQLearning', 'SARSALambda', 'SARSA', 'ExpectedSARSA']
    agent_pairs = [(a, b) for a in available_agents for b in available_agents]

    # for agents_name in agent_pairs:
    #     print(f"Training agents: {agents_name}")
    #     training(agents_name)

    show_game(random.choice(agent_pairs))
