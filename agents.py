import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
from collections import deque
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict

# torch.manual_seed(1234)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# possible inputs for the game
INPUTS = (
    (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1),
    (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1),
    (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 0, 1, 1),
    # (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1)
)


class Agent:
    def __init__(self, train=True, 
                epsilon=1.0, epsilon_min=0.0001, epsilon_decay_rate = 0.9999,
                gamma=0.99,
                alpha=0.0001, alpha_decay=0.01,
                batch_size=64
                ):
        
        self.frame_shape = (84,84)
        self.stack_size = 4
        self.buffer_size = 40000

        self.epsilon = epsilon  # exploration probability
        self.epsilon_decay_rate = epsilon_decay_rate # epsilon decay rate
        self.epsilon_min = epsilon_min  # minimum value of epsilon
        self.gamma = gamma  # discount
        self.alpha = alpha  # learning rate
        self.alpha_decay = alpha_decay  # learning rate decay
        self.batch_size = batch_size
        self.train_mode = train

        # initialize replay buffer
        storage = LazyTensorStorage(max_size=self.buffer_size, device="cpu")
        self.buffer = ReplayBuffer(storage=storage, batch_size=self.batch_size)
        
        self.model = self.__build_model()
        
        if self.train_mode:
            self.model.train()
            self.optimizer = Adam(params=self.model.parameters(), lr = self.alpha)
            self.loss_fn = nn.MSELoss()
        else:
            self.model.eval()

        # Set up observation window for inference
        self.frame_stack = deque(maxlen=self.stack_size)
        for _ in range(self.stack_size):
            self.frame_stack.append(np.random.rand(*self.frame_shape))


    def __build_model(self):
        return nn.Sequential(
            nn.Conv2d(self.stack_size, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7*7*64, 256),
            nn.ReLU(),
            nn.Linear(256, len(INPUTS))
            ).to(DEVICE)


    def remember(self, state, action, reward, done):
        current_stack = np.stack(self.frame_stack) 
        self.frame_stack.append(state)
        next_stack = np.stack(self.frame_stack)

        if self.train_mode:
            # Store the 4-frame stacks in the buffer
            data = TensorDict({
                "state": torch.as_tensor(current_stack, dtype=torch.uint8),
                "action": torch.as_tensor(action, dtype=torch.int8),
                "reward": torch.as_tensor(reward),
                "next_state": torch.as_tensor(next_stack, dtype=torch.uint8),
                "done": torch.as_tensor(done, dtype=torch.bool)
            }, batch_size=[])
        
            self.buffer.add(data)


    def act(self):
        # Perform epsilon greedy step
        # EPLORE
        if self.train_mode and np.random.rand() < self.epsilon:
            
            action_idx = np.random.randint(0, len(INPUTS))

        # EXPLOIT
        else:
            with torch.no_grad():
                state = np.stack(self.frame_stack)
                state_tensor = torch.from_numpy(state[None, :, :, :]).float().to(DEVICE)
                q_values = self.model(state_tensor)
                action_idx = torch.argmax(q_values).item()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay_rate

        return np.array(INPUTS[action_idx], dtype=np.int8), action_idx
    

    def train(self):
        if not self.train_mode or len(self.buffer) < self.batch_size:
                return
            
        batch = self.buffer.sample().to(DEVICE)
        state = batch["state"].float() / 255.0
        next_state = batch["next_state"].float() / 255.0

        q_next = self.model(next_state) 
        max_q_next = torch.max(q_next, dim=-1)[0]    
        
        y = torch.where(batch["done"], 
                        batch["reward"], 
                        batch["reward"] + self.gamma * max_q_next)
        
        q_pred = self.model(state)
        q_pred = q_pred.gather(1, batch["action"].long().unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q_pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def save(self, fname = 'model'):
        file = Path(__file__).parent / 'saved_models' / f'{fname}.pt'

        print(f"Saving model to: {file}")
        torch.save(self.model.state_dict(), file)


    def load(self, fname = 'model'):
        file = Path(__file__).parent / 'saved_models' / f'{fname}.pt'
        
        if file.exists():
            print(f"Loading model state dict from: {file}")
            self.model.load_state_dict(torch.load(file))
