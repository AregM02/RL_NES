import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from agents import Agent
from tcp_conncection.environment import FceuxEnv
from utils import animate_images


def main():
    N_episodes = 500
    max_steps = 10000
    skip_steps_train = 4
    stack_size = 4
    reward_history = []
    # frames = []

    # Create the agent and environment.
    # Make sure to keep the agent first, since PyTorch deadlocks sockets during CUDA runtime initialization.
    agent = Agent(train=True, stack_size=stack_size, epsilon=0.18289766427883497)
    agent.load()
    env = FceuxEnv()
    

    for ep in range(N_episodes):
        print(f"Episode: {ep+1}")
        state = env.reset() # reset to get the initial state
        state_stack = deque([state] * stack_size, maxlen=stack_size)

        episode_over = False
        total_reward = 0
        n_steps = 0

        while not episode_over and n_steps<max_steps:
            # Get an action for the current state
            current_stack = np.stack(state_stack)
            action, action_id = agent.act(current_stack)

            # Take the action
            next_state, reward, terminated = env.step(action)
            # Update observation stack
            state_stack.append(next_state)
            next_stack = np.stack(state_stack)
            
            # Remember experience and train the model
            agent.remember(current_stack, action_id, reward, next_stack, terminated)
            if n_steps % skip_steps_train == 0:
                agent.train()

            total_reward += reward
            n_steps += 1
            episode_over = terminated
            reward_history.append(reward)
            # frames.append(state)

        print(f"Total reward for the episode: {total_reward}. Current eps: {agent.epsilon}")
        agent.save()

    # animate_images(frames)
    # plt.plot(reward_history)
    # plt.xlabel('Steps')
    # plt.ylabel('Reward')
    # plt.savefig("reward_hist.pdf", dpi = 300)
    # plt.show()

    env.close()


main()