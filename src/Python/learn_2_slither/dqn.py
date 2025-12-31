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

from Python.constants.constants import (
    ACTION_SIZE,
    DQN_BATCH_SIZE,
    DQN_GAMMA,
    DQN_LEARNING_RATE,
    DQN_MEMORY_SIZE,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    HIDDEN_SIZE,
    LR_SCHEDULER_GAMMA,
    LR_SCHEDULER_STEP_SIZE,
    STATE_SIZE,
    TARGET_NETWORK_UPDATE_FREQ,
)
from Python.snake_game import Direction


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    """Deep Q-Network with single hidden layer architecture.

    Architecture: input -> hidden (ReLU) -> output
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class DQNAgent:
    """DQN Agent with experience replay and target network"""

    def __init__(
        self,
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        hidden_size=HIDDEN_SIZE,
        lr=DQN_LEARNING_RATE,
        gamma=DQN_GAMMA,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        batch_size=DQN_BATCH_SIZE,
        memory_size=DQN_MEMORY_SIZE,
        target_update=TARGET_NETWORK_UPDATE_FREQ,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.n_games = 0

        self.target_update = target_update
        self.steps = 0

        self.last_loss = 0.0

        # Apple tracking - keep previous closest apple to avoid confusion on ties
        self.closest_apple_idx = 0
        # Memory of discovered green apples (line-of-sight only)
        self.known_green_apples: set[tuple[int, int]] = set()

        self.device = get_device()

        # Policy network and target network
        self.policy_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net = DQN(state_size, hidden_size, action_size).to(self.device)
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

        point_l, point_r, point_u = self._get_relative_points(head, direction)
        danger_straight, danger_right, danger_left = self._get_danger_features(
            game, point_l, point_r, point_u
        )
        dir_l, dir_r, dir_u, dir_d = self._get_direction_onehot(direction)
        self._discover_green_apples(game, head)
        food_left, food_right, food_up, food_down = self._get_food_features(head)
        red_straight, red_right_pos, red_left_pos = self._get_red_apple_features(
            game, point_l, point_r, point_u
        )
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

    def _get_danger_features(self, game, point_l, point_r, point_u):
        """Check for collisions in straight, right, and left directions.

        Parameters
        ----------
        game : SnakeGame
            The game instance.
        point_l : tuple[int, int]
            Point directly ahead (straight).
        point_r : tuple[int, int]
            Point to the right.
        point_u : tuple[int, int]
            Point to the left.

        Returns
        -------
        tuple[int, int, int]
            Danger flags [danger_straight, danger_right, danger_left].
        """
        danger_straight = self._is_collision(game, point_l)
        danger_right = self._is_collision(game, point_r)
        danger_left = self._is_collision(game, point_u)
        return danger_straight, danger_right, danger_left

    def _is_collision(self, game, point):
        """Check if a point results in collision with wall or snake body"""
        if (
            point[0] < 0
            or point[0] >= game.width
            or point[1] < 0
            or point[1] >= game.height
        ):
            return 1
        if point in game.snake:
            return 1
        return 0

    def _get_relative_points(self, head, direction):
        """Get the three check points (straight, right, left) relative to current direction.

        Parameters
        ----------
        head : tuple[int, int]
            Current head position (x, y).
        direction : tuple[int, int]
            Direction vector (dx, dy).

        Returns
        -------
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
            Points to check for (straight, right, left) collisions.
        """
        dx, dy = direction

        if direction == Direction.LEFT.value:
            return (
                (head[0], head[1] - 1),
                (head[0] - 1, head[1]),
                (head[0] + 1, head[1]),
            )
        elif direction == Direction.RIGHT.value:
            return (
                (head[0], head[1] + 1),
                (head[0] + 1, head[1]),
                (head[0] - 1, head[1]),
            )
        elif direction == Direction.UP.value:
            return (
                (head[0] - 1, head[1]),
                (head[0], head[1] + 1),
                (head[0], head[1] - 1),
            )
        else:  # Direction.DOWN.value
            return (
                (head[0] + 1, head[1]),
                (head[0], head[1] - 1),
                (head[0], head[1] + 1),
            )

    def _discover_green_apples(self, game, head):
        """Discover green apples along cardinal directions and update memory.

        Scans up, down, left, and right from head position until board edges.
        Updates self.known_green_apples with newly visible apples.

        Parameters
        ----------
        game : SnakeGame
            The game instance.
        head : tuple[int, int]
            Current head position (x, y).
        """
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

    def _get_food_features(self, head):
        """Get one-hot encoding of green apple position relative to head.

        Picks the closest remembered green apple and returns whether it's
        to the left, right, up, or down from the head.

        Parameters
        ----------
        head : tuple[int, int]
            Current head position (x, y).

        Returns
        -------
        tuple[int, int, int, int]
            One-hot encoding [food_left, food_right, food_up, food_down].
        """
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

        return food_left, food_right, food_up, food_down

    def _get_red_apple_features(self, game, point_l, point_r, point_u):
        """Get one-hot encoding of red apple position adjacent to head.

        Only detects red apple if it's immediately adjacent (1 step away)
        in straight, right, or left directions.

        Parameters
        ----------
        game : SnakeGame
            The game instance.
        point_l : tuple[int, int]
            Point directly ahead (straight).
        point_r : tuple[int, int]
            Point to the right.
        point_u : tuple[int, int]
            Point to the left.

        Returns
        -------
        tuple[int, int, int]
            One-hot encoding [red_straight, red_right_pos, red_left_pos].
        """
        red_apple = game.red_apple
        red_straight = 1 if red_apple and red_apple == point_l else 0
        red_right_pos = 1 if red_apple and red_apple == point_r else 0
        red_left_pos = 1 if red_apple and red_apple == point_u else 0
        return red_straight, red_right_pos, red_left_pos

    def _get_direction_onehot(self, direction):
        """Get one-hot encoding of the current direction.

        Parameters
        ----------
        direction : tuple[int, int]
            Direction vector (dx, dy).

        Returns
        -------
        tuple[int, int, int, int]
            One-hot encoding [dir_left, dir_right, dir_up, dir_down].
        """
        dir_l = 1 if direction == Direction.LEFT.value else 0
        dir_r = 1 if direction == Direction.RIGHT.value else 0
        dir_u = 1 if direction == Direction.UP.value else 0
        dir_d = 1 if direction == Direction.DOWN.value else 0
        return dir_l, dir_r, dir_u, dir_d

    def get_action(self, state):
        """Epsilon-greedy action selection (matching nn_test.py approach)"""
        # TODO check if better way to decay
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

    def _sample_batch(self):
        """Sample a batch from memory.

        Returns
        -------
        list
            Batch of (state, action, reward, next_state, done) tuples.
        """
        if len(self.memory) > self.batch_size:
            return random.sample(self.memory, self.batch_size)
        return list(self.memory)

    def _prepare_tensors(self, mini_sample):
        """Convert batch data to tensors on device.

        Parameters
        ----------
        mini_sample : list
            List of (state, action, reward, next_state, done) tuples.

        Returns
        -------
        tuple
            (states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)
        """
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        states_tensor = torch.FloatTensor(np.array(states, dtype=np.float32)).to(
            self.device
        )
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_array = np.array(rewards, dtype=np.float32)
        rewards_tensor = torch.FloatTensor(rewards_array / 10.0).to(self.device)
        next_states_tensor = torch.FloatTensor(
            np.array(next_states, dtype=np.float32)
        ).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        return (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
        )

    def _compute_current_q(self, states_tensor, actions_tensor):
        """Compute Q values for taken actions.

        Parameters
        ----------
        states_tensor : torch.Tensor
            Batch of states.
        actions_tensor : torch.Tensor
            Batch of actions taken.

        Returns
        -------
        torch.Tensor
            Q values for the taken actions.
        """
        return (
            self.policy_net(states_tensor)
            .gather(1, actions_tensor.unsqueeze(1))
            .squeeze(1)
        )

    def _compute_target_q(self, rewards_tensor, next_states_tensor, dones_tensor):
        """Compute target Q values using target network.

        Parameters
        ----------
        rewards_tensor : torch.Tensor
            Batch of immediate rewards.
        next_states_tensor : torch.Tensor
            Batch of next states.
        dones_tensor : torch.Tensor
            Batch of done flags.

        Returns
        -------
        torch.Tensor
            Target Q values.
        """
        with torch.no_grad():
            next_q = self.target_net(next_states_tensor).max(1)[0]
            return rewards_tensor + (1 - dones_tensor) * self.gamma * next_q

    def _optimize_step(self, current_q, target_q):
        """Perform gradient descent step.

        Parameters
        ----------
        current_q : torch.Tensor
            Current Q values.
        target_q : torch.Tensor
            Target Q values.
        """
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.last_loss = loss.item()
        self.steps += 1

    def _update_target_network(self):
        """Update target network if enough steps have passed."""
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def replay(self):
        """Train on a batch of experiences (long memory training)."""
        if len(self.memory) == 0:
            return

        mini_sample = self._sample_batch()
        (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
        ) = self._prepare_tensors(mini_sample)

        current_q = self._compute_current_q(states_tensor, actions_tensor)
        target_q = self._compute_target_q(
            rewards_tensor, next_states_tensor, dones_tensor
        )

        self._optimize_step(current_q, target_q)
        self._update_target_network()

    def train_long_memory(self):
        """Alias for replay() - called after episode ends"""
        self.replay()
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
