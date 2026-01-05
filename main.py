import matplotlib.pyplot as plt
from agents import Agent
from tcp_conncection.environment import FceuxEnv
from utils import animate_images
import time


def main():
    # Create the agent and environment.
    # Make sure to keep the agent first, since PyTorch deadlocks sockets
    # during CUDA runtime initialization.
    agent = Agent(train=True)
    agent.load()
    env = FceuxEnv()
    
    N_episodes = 1000
    max_steps = 2000
    skip_steps_train = 4
    reward_history = []
    # frames = []

    for ep in range(N_episodes):
        print(f"Episode: {ep+1}")
        state = env.reset() # reset to get the initial state
        time.sleep(0.5)

        episode_over = False
        total_reward = 0
        n_steps = 0

        while not episode_over and n_steps<max_steps:
            # Get an action for the current state
            action, action_id = agent.act()

            # Take the action
            next_state, reward, terminated = env.step(action)

            # Remember experience and train the model
            agent.remember(state, action_id, reward, terminated)
            if n_steps % skip_steps_train == 0:
                agent.train()

            # Move to next state
            state = next_state

            total_reward += reward
            n_steps += 1
            episode_over = terminated
            reward_history.append(reward)
            # frames.append(state)

        print(f"Total reward for the episode: {total_reward}. Current eps: {agent.epsilon}")
        agent.save()

    # animate_images(frames,2)
    plt.plot(reward_history)
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.savefig("reward_hist.pdf", dpi = 300)
    # plt.show()

    env.close()


main()