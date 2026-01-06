# Display Settings

SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 900
SCREEN_OFFSET_Y = 100
SCREEN_OFFSET_X = 100
PVE_OFFSET_X = 50
SPRITE_SIZE = 32

GAME_GRID_SIZE = 10

GAME_SPEED = 10
DRAW_GRID = False

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
    "/home/luis/proyects/Learn2Slither/assets/images/background/background.jpg"
)

FONT = "/home/luis/proyects/Learn2Slither/assets/fonts/PressStart2P-Regular.ttf"

FOOD_SPRITE_PATH = "/home/luis/proyects/Learn2Slither/assets/images/food/red_apple.png"
BAD_FOOD_SPRITE_PATH = "/home/luis/proyects/Learn2Slither/assets/images/food/bomb.png"

SNAKE_BODY_HORIZONTAL_00 = (
    "/home/luis/proyects/Learn2Slither/assets/images/body/snake_body32_horizontal00.png"
)
SNAKE_BODY_HORIZONTAL_01 = (
    "/home/luis/proyects/Learn2Slither/assets/images/body/snake_body32_horizontal01.png"
)
SNAKE_BODY_VERTICAL_00 = (
    "/home/luis/proyects/Learn2Slither/assets/images/body/snake_body32_vertical00.png"
)
SNAKE_BODY_VERTICAL_01 = (
    "/home/luis/proyects/Learn2Slither/assets/images/body/snake_body32_vertical01.png"
)
SNAKE_BODY_LEFT_CONNECTOR = (
    "/home/luis/proyects/Learn2Slither/assets/images/body/snake_body_corner_00.png"
)
SNAKE_BODY_RIGHT_CONNECTOR = (
    "/home/luis/proyects/Learn2Slither/assets/images/body/snake_body_corner_01.png"
)
SNAKE_BODY_CORNER_02 = (
    "/home/luis/proyects/Learn2Slither/assets/images/body/snake_body_corner_02.png"
)
SNAKE_BODY_CORNER_03 = (
    "/home/luis/proyects/Learn2Slither/assets/images/body/snake_body_corner_03.png"
)
SNAKE_HEAD_00 = "/home/luis/proyects/Learn2Slither/assets/images/head/snake00.png"
SNAKE_HEAD_01 = "/home/luis/proyects/Learn2Slither/assets/images/head/snake01.png"
SNAKE_TAIL_00 = "/home/luis/proyects/Learn2Slither/assets/images/tail/tail_final00.png"
SNAKE_TAIL_01 = "/home/luis/proyects/Learn2Slither/assets/images/tail/tail_final01.png"
SNAKE_TAIL_02 = "/home/luis/proyects/Learn2Slither/assets/images/tail/tail_final02.png"
SNAKE_TAIL_03 = "/home/luis/proyects/Learn2Slither/assets/images/tail/tail_final03.png"
SNAKE_TAIL_04 = "/home/luis/proyects/Learn2Slither/assets/images/tail/tail_final04.png"

# Wall Sprites
WALL_CORNER = (
    "/home/luis/proyects/Learn2Slither/assets/images/background/walls/corner.png"
)
WALL_CENTER_HORI = (
    "/home/luis/proyects/Learn2Slither/assets/images/background/walls/center_hori.png"
)
WALL_CENTER_VER = (
    "/home/luis/proyects/Learn2Slither/assets/images/background/walls/center_ver.png"
)
WALL_LEFT_END = (
    "/home/luis/proyects/Learn2Slither/assets/images/background/walls/left_end.png"
)
WALL_RIGHT_END = (
    "/home/luis/proyects/Learn2Slither/assets/images/background/walls/right_end.png"
)
WALL_TOP_END = (
    "/home/luis/proyects/Learn2Slither/assets/images/background/walls/top_end.png"
)
WALL_BOT_END = (
    "/home/luis/proyects/Learn2Slither/assets/images/background/walls/bot_end.png"
)

# Ground Tiles
GROUND_GRASS_BLUR = (
    "/home/luis/proyects/Learn2Slither/assets/images/background/grass_blur.png"
)
GROUND_GRASS_BLUR2 = (
    "/home/luis/proyects/Learn2Slither/assets/images/background/grass_blur2.png"
)
GROUND_SNOW_MIDDLE = (
    "/home/luis/proyects/Learn2Slither/assets/images/background/snow_middle.png"
)

# Game Configuration
MAX_STEPS_PER_EPISODE = 1000

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
