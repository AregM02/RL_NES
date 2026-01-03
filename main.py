import matplotlib.pyplot as plt
from agents import Agent
from tcp_conncection.environment import FceuxEnv
from utils import animate_images

def main():
    # Create the agent and environment.
    # Make sure to keep the agent first, since PyTorch deadlocks sockets
    # during CUDA runtime initialization.
    agent = Agent() 
    agent.load()
    env = FceuxEnv()
    
    N_episodes = 1  # total number of episodes
    step_history = []
    reward_history = []
    # frames = []

    for ep in range(N_episodes):
        print(f"Episode: {ep+1}")
        state = env.reset() # reset to get the initial state

        episode_over = False
        total_reward = 0
        n_steps = 0

        while not episode_over:
            # Get an action for the current state
            action, action_id = agent.act()

            # Take the action
            next_state, reward, terminated = env.step(action)

            # Remember experience and train the model
            agent.remember(state, action_id, reward, terminated)
            agent.train()

            # Move to next state
            state = next_state

            total_reward += reward
            n_steps += 1
            episode_over = terminated
            reward_history.append(reward)
            # frames.append(state)

        step_history.append(n_steps)
        reward_history.append(total_reward)

    # animate_images(frames, interval = 2)

    # plt.plot(step_history)
    # plt.xlabel('Episodes')
    # plt.ylabel('Number of Steps')
    # plt.show()

    # plt.plot(reward_history)
    # plt.xlabel('Episodes')
    # plt.ylabel('Reward')
    # plt.show()

    env.close()
    agent.save()

main()