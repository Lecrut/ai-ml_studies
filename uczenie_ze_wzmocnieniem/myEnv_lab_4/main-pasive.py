#%% Imports 
from labyrinthMAP import LabyrinthMap

#%% Show list of states and possible transitions
def print_states_and_transitions(env, states):
    for s in states:
        actions = env.get_possible_actions(s)
        for a in actions:
            next_states = env.get_next_states(s, a)
            print("State: " + str(s) + " action: " + str(a) + " " + "list of possible next states: ", next_states)



#%% Value Iteration Algorithm
def value_iteration(mdp, gamma, theta):
    V = dict()
    policy = dict()

    for current_state in mdp.get_all_states():
        V[current_state] = 0.0
        actions = mdp.get_possible_actions(current_state)
        policy[current_state] = actions[0] if len(actions) > 0 else None

    while True:
        delta = 0.0

        for s in mdp.get_all_states():
            v = V[s]
            best_value = None
            best_action = None

            for a in mdp.get_possible_actions(s):
                raw_next = mdp.get_next_states(s, a)
                next_items = []
                if isinstance(raw_next, dict):
                    next_items = list(raw_next.items())
                elif isinstance(raw_next, (list, tuple)):
                    if len(raw_next) == 0:
                        next_items = [(s, 1.0)]   
                    else:
                        prob = 1.0 / len(raw_next)
                        next_items = [(ns, prob) for ns in raw_next]
                else:
                    next_items = [(raw_next, 1.0)]

                sum_value = 0.0
                for next_state, prob in next_items:
                    r = mdp.get_reward(s, a, next_state)
                    sum_value += prob * (r + gamma * V.get(next_state, 0.0))

                if best_value is None or sum_value > best_value:
                    best_value = sum_value
                    best_action = a

            if best_value is None:
                best_value = v

            V[s] = best_value
            policy[s] = best_action
            delta = max(delta, abs(v - best_value))

        if delta < theta:
            break

    print(policy)

    return policy, V

#%% Show game
def show_game(env, policy, V, is_show_map=True):
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
            action = policy[current_state]

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
        print('None agent ended in less than 200 steps.')

#%% Play game
if __name__ == "__main__":
    env = LabyrinthMap(2)

    states = env.get_all_states()
    # print('States:', states)

    # print_states_and_transitions(env, states)

    gamma = 0.9
    theta = 0.001
    policy, V = value_iteration(env, gamma, theta)

    show_game(env, policy, V, is_show_map=True)
