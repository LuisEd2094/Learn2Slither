GAME_SPEED = 10

DARK_GREEN = [(0, 128, 0), (50, 205, 50), (144, 238, 144)]
LIGHT_GREEN = [(200, 255, 200), (180, 230, 180)]
YELLOW_ORANGE = [(255, 255, 0), (255, 165, 0), (255, 69, 0)]
LIGHT_BLUE = [(173, 216, 230), (135, 206, 250)]
DARK_BLUE = [(0, 0, 139), (25, 25, 112), (0, 0, 205)]
CYAN = [(0, 255, 255), (64, 224, 208)]

LIGHT_BLUE = [(173, 216, 230), (135, 206, 250)]
DARK_BLUE = [(0, 0, 139), (25, 25, 112), (0, 0, 205)]
CYAN = [(0, 255, 255), (64, 224, 208)]
LIGHT_PURPLE = [(216, 191, 216), (221, 160, 221)]
DARK_PURPLE = [(128, 0, 128), (75, 0, 130)]
LAVENDER = [(230, 230, 250), (238, 130, 238)]
LIGHT_GRAY = [(211, 211, 211), (192, 192, 192)]
DARK_GRAY = [(105, 105, 105), (64, 64, 64)]
BLACK_WHITE = [(0, 0, 0), (255, 255, 255)]


BACKGROUND_TILE = (
    "/home/luis/proyects/Learn2Slither/assets/images/background/flowers.png"
)

FONT = "/home/luis/proyects/Learn2Slither/assets/fonts/PressStart2P-Regular.ttf"


# Game Configuration
GAME_GRID_SIZE = 10
MAX_STEPS_PER_EPISODE = 1500

# Model Paths
DEFAULT_SAVE_PATH = "model/simple_dqn.pt"
DEFAULT_LOAD_PATH = None

# DQN Network Configuration
STATE_SIZE = 14
ACTION_SIZE = 3
HIDDEN_SIZE = 256

# DQN Training Parameters
DQN_LEARNING_RATE = 0.001
DQN_GAMMA = 0.9
DQN_BATCH_SIZE = 1000
DQN_MEMORY_SIZE = 100_000
TARGET_NETWORK_UPDATE_FREQ = 50

# Learning Rate Scheduler Parameters
LR_SCHEDULER_STEP_SIZE = 100
LR_SCHEDULER_GAMMA = 0.8

# Exploration Parameters
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995

# Rewards
REWARD_FOOD_EATEN = 10
REWARD_RED_APPLE_EATEN = -5
REWARD_DEATH = -10
REWARD_NEUTRAL = 0

GREEN_APPLE_TO_SPAWN = 2
