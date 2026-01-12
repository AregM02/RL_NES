import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
from torchrl.data import ReplayBuffer, LazyTensorStorage, PrioritizedSampler
from tensordict import TensorDict

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

INPUTS = (
    (0, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 1, 0, 1),
    (0, 1, 1, 0),
    (0, 1, 1, 1),
    (0, 0, 0, 1),
)

class Agent:
    def __init__(self, train=True, 
                epsilon=1.0, epsilon_min=0.01, epsilon_decay_rate = 0.99999,
                gamma=0.99,
                alpha=0.00025,
                batch_size=256,
                target_update_freq=1000, # sync every 1000 steps
                stack_size = 4, buffer_size = 500000,
                ):
        
        self.frame_shape = (88, 128)
        self.stack_size = stack_size
        self.buffer_size = buffer_size
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.train_mode = train
        self.target_update_freq = target_update_freq
        self.update_step = 0

        self.buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=self.buffer_size, device="cpu"),
                                   sampler=PrioritizedSampler(max_capacity=self.buffer_size, alpha=0.6, beta=0.4),
                                   batch_size=self.batch_size)
        
        # Build Online and Target Networks
        self.model = self.__build_model()
        self.target_model = self.__build_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() # target never trains via backprop
        
        if self.train_mode:
            self.model.train()
            self.optimizer = Adam(params=self.model.parameters(), lr = self.alpha)
            self.loss_fn = nn.HuberLoss() # More robust to outliers than MSE
        else:
            self.model.eval()


    def __build_model(self):
        return nn.Sequential(
            nn.Conv2d(self.stack_size, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 12, 512), # previously 7*7*64 for (88, 88)
            nn.ReLU(),
            nn.Linear(512, len(INPUTS))
            ).to(DEVICE)


    def remember(self, state, action, reward, next_state, done):
            # Store the ACTUAL stacks as they were seen
            if self.train_mode:
                data = TensorDict({
                    "state": torch.as_tensor(state, dtype=torch.uint8),
                    "action": torch.as_tensor(action, dtype=torch.int8),
                    "reward": torch.as_tensor(reward, dtype=torch.float32),
                    "next_state": torch.as_tensor(next_state, dtype=torch.uint8),
                    "done": torch.as_tensor(done, dtype=torch.bool)
                }, batch_size=[])
                self.buffer.add(data)


    def act(self, state_stack):
        if self.train_mode and np.random.rand() < self.epsilon:
            action_idx = np.random.randint(0, len(INPUTS))
            print(action_idx)
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state_stack[None, :, :, :]).float().to(DEVICE) / 255.0
                q_values = self.model(state_tensor)
                action_idx = torch.argmax(q_values).item()

        if self.train_mode:
            self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.epsilon_min)

        return np.array(INPUTS[action_idx], dtype=np.int8), action_idx
    

    def train(self):
        if not self.train_mode or len(self.buffer) < self.batch_size:
                return
            
        batch, info = self.buffer.sample(return_info=True)
        batch = batch.to(DEVICE)
        state = batch["state"].float() / 255.0
        next_state = batch["next_state"].float() / 255.0

        # --- DOUBLE DQN LOGIC ---
        with torch.no_grad():
            # 1. Online model selects the best action
            next_q_online = self.model(next_state)
            best_next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            
            # 2. Target model evaluates that action
            next_q_target = self.target_model(next_state)
            max_q_next = next_q_target.gather(1, best_next_actions).squeeze(1)

        y = torch.where(batch["done"], 
                        batch["reward"], 
                        batch["reward"] + self.gamma * max_q_next)
        
        q_pred = self.model(state).gather(1, batch["action"].long().unsqueeze(1)).squeeze(1)

        # update sampler priority of the buffer
        td_error = torch.abs(y - q_pred).detach()
        self.buffer.update_priority(info["index"], td_error)

        # standard backprop
        loss = self.loss_fn(q_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update Target Network
        self.update_step += 1
        if self.update_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())


    def save(self, fname = 'checkpoint'):
        file = Path(__file__).parent / 'saved_models' / f'{fname}.pt'
        
        print(f"Saving checkpoint to: {file}")
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, file)


    def load(self, fname = 'checkpoint'):
        file = Path(__file__).parent / 'saved_models' / f'{fname}.pt'

        if file.exists():
            checkpoint = torch.load(file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.train_mode:
                self.target_model.load_state_dict(self.model.state_dict())
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # kinda necessary
            print(f"Loaded previous checkpoint from: {file}")
