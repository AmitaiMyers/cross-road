import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # --- HYPERPARAMETERS ---
        # These control how fast/well the AI learns
        self.gamma = 0.95  # Discount Factor (Care about future)
        self.epsilon = 1.0  # Exploration Rate (Start at 100% random)
        self.epsilon_min = 0.01  # Minimum exploration (1% random)
        self.epsilon_decay = 0.995  # How fast to stop exploring
        self.learning_rate = 0.001
        self.batch_size = 32  # How many memories to learn from at once

        # --- Memory ---
        self.memory = deque(maxlen=2000)

        # --- Brain ---
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)

        return torch.argmax(q_values).item()

    def replay(self):
        # 1. Check if we have enough memory
        if len(self.memory) < self.batch_size:
            return

        # 2. Sample a random minibatch
        minibatch = random.sample(self.memory, self.batch_size)

        # --- VECTORIZATION (The Speed Boost) ---
        # Instead of looping, we convert the lists into big PyTorch Tensors
        # This lets the GPU/CPU do all the math in ONE go.

        # Stack all states into a single tensor of shape [32, 7]
        states = torch.FloatTensor(np.array([i[0] for i in minibatch]))

        # Stack actions [32, 1]
        actions = torch.LongTensor([i[1] for i in minibatch]).unsqueeze(1)

        # Stack rewards [32]
        rewards = torch.FloatTensor([i[2] for i in minibatch])

        # Stack next states [32, 7]
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch]))

        # Stack done flags [32]
        dones = torch.FloatTensor([i[4] for i in minibatch])

        # 3. Current Q Values (Prediction)
        # We pass ALL 32 states to the model at once.
        # gather(1, actions) selects only the Q-value for the action we actually took.
        current_q_values = self.model(states).gather(1, actions).squeeze(1)

        # 4. Next Q Values (Target)
        # We predict the future for ALL 32 next_states at once.
        next_q_values = self.model(next_states).detach()
        # Take the maximum Q-value for the next step
        max_next_q = next_q_values.max(1)[0]

        # Bellman Equation (Vectorized)
        # If done is 1, the (1-done) term becomes 0, ignoring the future.
        expected_q_values = rewards + (self.gamma * max_next_q * (1 - dones))

        # 5. Backpropagation (ONE UPDATE ONLY)
        self.optimizer.zero_grad()
        loss = self.criterion(current_q_values, expected_q_values)
        loss.backward()
        self.optimizer.step()


    def save(self, filename):
        """Saves the neural network weights to a file."""
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        """Loads the neural network weights from a file."""
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()  # Set to evaluation mode (optional, good practice)