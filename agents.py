import numpy as np

class Agent:
    def __init__(self):
        pass

    def get_action(self, state):
        action = np.array([0,1,1,0], dtype=np.uint8)
        return action
    
    def decay_epsilon(self):
        pass

    def update(self, state, action, reward, terminated, next_state):
        pass