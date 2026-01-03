import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
from collections import deque

torch.manual_seed(1234)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# possible inputs for the game
INPUTS = (
    (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1),
    (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1),
    (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 0, 1, 1),
    (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1)
)


class Agent:
    def __init__(self, train=True, epsilon_decay_rate = 0.9999, epsilon=1.0,
                 epsilon_min=0.001, gamma=0.95, alpha=0.001, alpha_decay=0.01,
                 batch_size=32):
        
        self.frame_shape = (84,84)
        self.stack_size = 4

        self.epsilon = epsilon  # exploration probability
        self.epsilon_decay_rate = epsilon_decay_rate # epsilon decay rate
        self.epsilon_min = epsilon_min  # minimum value of epsilon
        self.gamma = gamma  # discount
        self.alpha = alpha  # learning rate
        self.alpha_decay = alpha_decay  # learning rate decay
        self.batch_size = batch_size
        self.train_mode = train
        self.buffer = ReplayBuffer(capacity=50000,
                                   frame_shape=self.frame_shape,
                                   stack_size=self.stack_size) # replay buffer
        
        self.model = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
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
        
        if self.train_mode:
            self.model.train()
            self.optimizer = Adam(params=self.model.parameters(),
                              lr = self.alpha, weight_decay=self.alpha_decay)
            self.loss_fn = nn.MSELoss()
        else:
            self.model.eval()

        # Set up observation window for inference
        self.frame_stack = deque(maxlen=4)
        for _ in range(self.stack_size):
            self.frame_stack.append(np.random.rand(*self.frame_shape))


    def save(self, fname = 'model'):
        file = Path(__file__).parent / 'saved_models' / f'{fname}.pt'

        print(f"Saving model to: {file}")
        torch.save(self.model.state_dict(), file)


    def load(self, fname = 'model'):
        file = Path(__file__).parent / 'saved_models' / f'{fname}.pt'
        
        if file.exists():
            print(f"Loading model state dict from: {file}")
            self.model.load_state_dict(torch.load(file))


    def remember(self, state, action, reward, done):
        # save the experience to the replay buffer
        self.buffer.add(state, action, reward, done)

        # also add to the online frame_stack for inference
        state = state.astype(np.float32) / 255.0
        self.frame_stack.append(state)


    def act(self):
        if self.train_mode and np.random.rand() < self.epsilon:
            # perform epsilon greedy
            action_idx = np.random.randint(0, len(INPUTS))

        else:
            with torch.no_grad():
                state = np.stack(self.frame_stack)
                state_tensor = torch.from_numpy(state[None, :, :, :]).float().to(DEVICE)
                q_values = self.model(state_tensor)
                action_idx = torch.argmax(q_values).item()

        return np.array(INPUTS[action_idx], dtype=np.int8), action_idx
    

    def decay_eps(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay_rate


    def train(self):
        # minibatch of the follwing format: (s_t, a_t, r_t, s_{t+1}, done)
        minibatch = self.buffer.sample(self.batch_size)

        if minibatch is None:
            return
    
        if len(self.buffer) < self.batch_size:
            return

        # define targets
        q_next = self.model(minibatch[3]) # (batch, len(INPUTS))
        max_q_next = torch.max(q_next, dim=-1)[0]    
        y = torch.where(minibatch[-1], # dones
                        minibatch[2], # rewards if done
                        minibatch[2] + self.gamma * max_q_next) # Q targets
        
        q_pred = self.model(minibatch[0])
        q_pred = q_pred.gather(1, minibatch[1].long().unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q_pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.decay_eps()


class ReplayBuffer:
    def __init__(self, capacity, frame_shape=(84, 84), stack_size=4):
        self.capacity = capacity
        self.stack_size = stack_size

        self.frames = np.zeros((capacity, *frame_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

        self.idx = 0  # next index to insert
        self.size = 0  # current buffer size


    def add(self, frame, action, reward, done):
        """Add a single transition to the buffer."""
        self.frames[self.idx] = frame
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


    def _valid_index(self, idx):
        """Check that idx can be used for stacking frames."""
        if self.size < self.stack_size:
            return False

        # Compute indices of the stack
        indices = [(idx - offset) % self.capacity for offset in reversed(range(self.stack_size))]
        
        # Cannot cross episode boundaries
        if np.any(self.dones[indices[:-1]]):
            return False
        
        return True


    def _get_stack(self, idx):
        """Return a stacked state s_t of shape (stack_size, H, W)."""
        assert self._valid_index(idx), f"Invalid index {idx} for stack"
        indices = [(idx - offset) % self.capacity for offset in reversed(range(self.stack_size))]
        return self.frames[indices]


    def sample(self, batch_size):
        """Sample a batch of transitions (s_t, a_t, r_t, s_{t+1}, done)."""
        if self.size < self.stack_size:
            return None

        # Candidate indices: any index that could form a full stack
        candidates = np.arange(self.size)  # full buffer
        mask = np.array([self._valid_index(i) for i in candidates])
        valid_indices = candidates[mask]

        if len(valid_indices) < batch_size:
            return None  # not enough valid stacks yet

        # Sample batch_size indices
        idxs = np.random.choice(valid_indices, size=batch_size, replace=False)

        # Construct batches
        states = np.stack([self._get_stack(i) for i in idxs]).astype(np.float32) / 255.0
        next_states = np.stack([self._get_stack((i + 1) % self.capacity) for i in idxs]).astype(np.float32) / 255.0
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        dones = self.dones[idxs]

        # Convert to torch tensors if desired
        states = torch.from_numpy(states).to(DEVICE)  # (batch, stack, H, W)
        next_states = torch.from_numpy(next_states).to(DEVICE)
        actions = torch.from_numpy(actions).to(DEVICE)
        rewards = torch.from_numpy(rewards).to(DEVICE)
        dones = torch.from_numpy(dones).to(DEVICE)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size
