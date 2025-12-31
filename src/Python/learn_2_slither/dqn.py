"""
Clean PyTorch DQN Agent for Snake Q-Learning
Full vision of grid, simple state representation
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Python.constants.constants import LR_SCHEDULER_GAMMA, LR_SCHEDULER_STEP_SIZE
from Python.snake_game import Direction


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    """Deep Q-Network with configurable architecture"""

    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        # hidden_sizes can be int (simple - 1 hidden layer) or list (variable depth)
        if isinstance(hidden_sizes, int):
            # Single hidden layer to match nn_test.py: input -> hidden -> output
            self.linear1 = nn.Linear(input_size, hidden_sizes)
            self.linear2 = nn.Linear(hidden_sizes, output_size)
            self.layers = None
        else:
            # Multiple hidden layers (list mode)
            layers = []
            prev_size = input_size

            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, output_size))
            self.layers = nn.ModuleList(layers)

    def forward(self, x):
        if self.layers is None:
            # Single hidden layer path (matches nn_test.py)
            x = torch.relu(self.linear1(x))
            x = self.linear2(x)
        else:
            # Multiple hidden layers path
            for layer in self.layers[:-1]:
                x = torch.relu(layer(x))
            x = self.layers[-1](x)
        return x


class DQNAgent:
    """DQN Agent with experience replay and target network"""

    def __init__(
        self,
        state_size,
        action_size=3,
        hidden_size=256,
        hidden_sizes=None,
        lr=0.001,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995,
        batch_size=1000,
        memory_size=100_000,
        target_update=5,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.n_games = 0  # Track number of games for epsilon calculation

        self.target_update = target_update
        self.steps = 0

        # Loss tracking
        self.last_loss = 0.0

        # Apple tracking - keep previous closest apple to avoid confusion on ties
        self.closest_apple_idx = 0  # Track index of closest apple

        # Movement tracking
        self.head_history = []  # Track head positions for movement variance
        self.last_moves = []  # Track last 3 actions
        self.size_history = []  # Track size over time
        # Memory of discovered green apples (line-of-sight only)
        self.known_green_apples: set[tuple[int, int]] = set()

        self.device = get_device()

        # Use hidden_sizes if provided, otherwise use hidden_size
        arch = hidden_sizes if hidden_sizes is not None else hidden_size

        # Policy network and target network
        self.policy_net = DQN(state_size, arch, action_size).to(self.device)
        self.target_net = DQN(state_size, arch, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size)

        # Learning rate scheduler to reduce LR over time
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=LR_SCHEDULER_STEP_SIZE,
            gamma=LR_SCHEDULER_GAMMA,
        )
        self.current_lr = lr

    def get_state(self, game):
        """Extract state with 14 features

        Features:
        - Danger straight/right/left (relative to direction): 3 features
        - Current direction one-hot (left/right/up/down): 4 features
        - Green apple location relative to head (left/right/up/down): 4 features
        - Red apple adjacent straight/right/left: 3 features
        Total: 14 features

        When multiple green apples exist, tracks the CLOSEST one.
        Red apple is only tracked if it's immediately adjacent (1 step away).
        """
        head = game.get_snake_head()
        direction = game.get_heading()

        # Direction vectors
        dir_map = {
            (0, -1): 2,
            (0, 1): 3,
            (-1, 0): 0,
            (1, 0): 1,
        }  # LEFT, RIGHT, UP, DOWN
        _ = dir_map.get(direction, 0)

        # Map to check points ahead based on direction
        # direction: (point_straight, point_right, point_left)
        if direction == (-1, 0):  # LEFT
            point_l = (head[0], head[1] - 1)  # straight
            point_r = (head[0] - 1, head[1])  # right (down)
            point_u = (head[0] + 1, head[1])  # left (up)
        elif direction == (1, 0):  # RIGHT
            point_l = (head[0], head[1] + 1)  # straight
            point_r = (head[0] + 1, head[1])  # right (up)
            point_u = (head[0] - 1, head[1])  # left (down)
        elif direction == (0, -1):  # UP
            point_l = (head[0] - 1, head[1])  # straight
            point_r = (head[0], head[1] + 1)  # right (right)
            point_u = (head[0], head[1] - 1)  # left (left)
        else:  # DOWN (0, 1)
            point_l = (head[0] + 1, head[1])  # straight
            point_r = (head[0], head[1] - 1)  # right (left)
            point_u = (head[0], head[1] + 1)  # left (right)

        # Check collision for straight, right, left
        danger_straight = self._is_collision(game, point_l)
        danger_right = self._is_collision(game, point_r)
        danger_left = self._is_collision(game, point_u)

        # Direction one-hot: [dir_left, dir_right, dir_up, dir_down]
        dir_l = 1 if direction == (-1, 0) else 0
        dir_r = 1 if direction == (1, 0) else 0
        dir_u = 1 if direction == (0, -1) else 0
        dir_d = 1 if direction == (0, 1) else 0

        # Visible green apples only along cross (up/down/left/right), not blocked by body
        current_apples = set(game.get_green_apples())
        # Drop eaten apples from memory
        self.known_green_apples &= current_apples

        dirs = [
            Direction.RIGHT.value,
            Direction.LEFT.value,
            Direction.DOWN.value,
            Direction.UP.value,
        ]
        for dx, dy in dirs:
            x, y = head
            while 0 <= x < game.width and 0 <= y < game.height:
                if (x, y) in current_apples:
                    self.known_green_apples.add((x, y))
                x += dx
                y += dy

        # Pick closest remembered green apple (if any)
        food = None
        if self.known_green_apples:
            distances = [
                abs(ax - head[0]) + abs(ay - head[1])
                for ax, ay in self.known_green_apples
            ]
            closest_idx = int(np.argmin(distances))
            food = list(self.known_green_apples)[closest_idx]

        if food:
            food_left = 1 if food[0] < head[0] else 0
            food_right = 1 if food[0] > head[0] else 0
            food_up = 1 if food[1] < head[1] else 0
            food_down = 1 if food[1] > head[1] else 0
        else:
            food_left = food_right = food_up = food_down = 0

        # Red apple adjacent detection (only if it's 1 step away)
        red_apple = game.red_apple
        red_straight = 1 if red_apple and red_apple == point_l else 0
        red_right_pos = 1 if red_apple and red_apple == point_r else 0
        red_left_pos = 1 if red_apple and red_apple == point_u else 0

        # Combine: 3 + 4 + 4 + 3 = 14 features
        state = [
            danger_straight,
            danger_right,
            danger_left,
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            food_left,
            food_right,
            food_up,
            food_down,
            red_straight,
            red_right_pos,
            red_left_pos,
        ]
        return np.array(state, dtype=np.float32)

    def _is_collision(self, game, point):
        """Check if a point results in collision with wall or snake body"""
        # Wall collision
        if (
            point[0] < 0
            or point[0] >= game.width
            or point[1] < 0
            or point[1] >= game.height
        ):
            return 1
        # Self collision
        if point in game.snake:
            return 1
        return 0

    def get_action(self, state):
        """Epsilon-greedy action selection (matching nn_test.py approach)"""
        # nn_test.py uses: epsilon = 80 - n_games, compared to random.randint(0, 200)
        # This means exploration probability starts at 80/200 = 40% and decreases
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        # Track action for movement history
        self.last_moves.append(action)
        if len(self.last_moves) > 10:
            self.last_moves.pop(0)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Train immediately on a single step (like reference implementation)"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        done_tensor = torch.FloatTensor([done]).to(self.device)

        # Current Q value
        current_q = (
            self.policy_net(state_tensor)
            .gather(1, action_tensor.unsqueeze(1))
            .squeeze(1)
        )

        # Target Q value
        with torch.no_grad():
            next_q = self.target_net(next_state_tensor).max(1)[0]
            target_q = reward_tensor + (1 - done_tensor) * self.gamma * next_q

        # Compute loss and update
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.last_loss = loss.item()
        self.steps += 1

    def replay(self):
        """Train on a batch of experiences (long memory training)"""
        if len(self.memory) == 0:
            return

        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        # Convert directly to tensors (states are already consistent size)
        states_tensor = torch.FloatTensor(np.array(states, dtype=np.float32)).to(
            self.device
        )
        actions_tensor = torch.LongTensor(actions).to(self.device)
        # Normalize rewards to prevent Q-value explosion
        rewards_array = np.array(rewards, dtype=np.float32)
        rewards_tensor = torch.FloatTensor(rewards_array / 10.0).to(
            self.device
        )  # Scale down rewards
        next_states_tensor = torch.FloatTensor(
            np.array(next_states, dtype=np.float32)
        ).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = (
            self.policy_net(states_tensor)
            .gather(1, actions_tensor.unsqueeze(1))
            .squeeze(1)
        )

        # Target Q values using target network
        with torch.no_grad():
            next_q = self.target_net(next_states_tensor).max(1)[0]
            target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q

        # Compute loss and update
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), 10.0
        )  # More aggressive clipping
        self.optimizer.step()

        self.last_loss = loss.item()
        self.steps += 1

        # Update target network periodically
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_long_memory(self):
        """Alias for replay() - called after episode ends"""
        self.replay()
        # Step the learning rate scheduler
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
            self.current_lr = self.optimizer.param_groups[0]['lr']

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Save model"""
        torch.save(
            {
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps': self.steps,
            },
            path,
        )

    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.steps = checkpoint.get('steps', 0)
