import matplotlib.pyplot as plt
from agents import Agent
from tcp_conncection.environment import FceuxEnv

def main():
    # Variables for training/testing
    episodes = 5  # total number of episodes
    max_steps = 100  # maximum number of steps per episode

    # Q-learning algorithm hyperparameters to tune
    alpha = 0.35  # learning rate: you may change it to see the difference
    gamma = 0.75  # discount factor: you may change it to see the difference

    # Exploration-exploitation trade-off
    epsilon = 1.0  # probability the agent will explore (initial value is 1.0)
    epsilon_min = 0.001  # minimum value of epsilon
    epsilon_decay = 0.9999  # decay multiplied with epsilon after each episode

    # Create Taxi environment
    env = FceuxEnv()
    agent = Agent()
    
    step_history = []
    reward_history = []

    for ep in range(episodes):
        print(f"Episode: {ep+1}")
        state = env.reset() # reset to get the initial state
        episode_over = False
        total_reward = 0
        n_steps = 0

        while not episode_over:
            # get an action for the current state
            action = agent.get_action(state)

            # Take the action
            next_state, reward, terminated = env.step(action)

            # Update knowledge
            agent.update(state, action, reward, terminated, next_state)

            # Move to next state
            state = next_state

            total_reward += reward
            n_steps += 1
            episode_over = terminated

        # reduce exploration rate
        agent.decay_epsilon()

        step_history.append(n_steps)
        reward_history.append(total_reward)

    plt.plot(step_history)
    plt.xlabel('Episodes')
    plt.ylabel('Number of Steps')
    plt.show()

    plt.plot(reward_history)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.show()

    env.close()


main()