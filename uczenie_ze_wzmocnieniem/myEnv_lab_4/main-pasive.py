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
    """
            This function calculate optimal policy for the specified MDP using Value Iteration approach:

            'mdp' - model of the environment, use following functions:
                get_all_states - return list of all states available in the environment
                get_possible_actions - return list of possible actions for the given state
                get_next_states - return list of possible next states with a probability for transition from state by taking
                                  action into next_state
                get_reward - return the reward after taking action in state and landing on next_state


            'gamma' - discount factor for MDP
            'theta' - algorithm should stop when minimal difference between previous evaluation of policy and current is
                      smaller than theta
            Function returns optimal policy and value function for the policy
       """

    V = dict()
    policy = dict()

    # init with a policy with first avail action for each state
    for current_state in mdp.get_all_states():
        V[current_state] = 0
        actions = mdp.get_possible_actions(current_state)
        policy[current_state] = actions[0] if actions else None
    
    while True:
        delta = 0
        for s in mdp.get_all_states():
            v = V[s]
            newV_s = 0
            results = []
            for a in mdp.get_possible_actions(s):
                sum_value = 0
                next_states = mdp.get_next_states(s, a)
                if not next_states:
                    reward = mdp.get_reward(s, a, s)  
                    sum_value = reward + gamma * V[s]
                else:
                    prob = 1 / len(next_states)
                    for next_state in next_states:
                        new_next_state = (next_state,) + s[1:]  
                        reward = mdp.get_reward(s, a, new_next_state)
                        sum_value += prob * (reward + (gamma * V[new_next_state]))
                results.append((sum_value, a))
            if results:
                newV_s, best_action = max(results)
            else:
                newV_s = V[s]  
                best_action = policy[s]  
            delta = max(delta, abs(v - newV_s))
            V[s] = newV_s
            policy[s] = best_action
        if delta < theta:   
            break
        
    return policy, V


#%% Show game
def show_game(env, V, is_show_map=True):
    env.reset()
    done = [False for _ in range(env._num_agents)]
    rewards = [0 for _ in range(env._num_agents)]
    steps = 0

    while not any(done) and steps < 100:
        steps += 1
        for i in range(env._num_agents):
            if done[i]:
                continue

            action = env.get_next_move(i, policy)
            _, reward, terminal = env.step(i, action)
            rewards[i] += reward

            if is_show_map:
                env.show_map()

            if terminal:
                done[i] = True
                print(f"Agent {i} finished with reward: {rewards[i]} in {steps} steps")
                break

    if not any(done):
        print('None agent ended in less than 100000 steps.')


#%% Play game
if __name__ == "__main__":
    env = LabyrinthMap(2)

    states = env.get_all_states()
    # print('States:', states)

    # print_states_and_transitions(env, states)

    gamma = 0.9
    theta = 0.0001  
    policy, V = value_iteration(env, gamma, theta)

    show_game(env, V)
