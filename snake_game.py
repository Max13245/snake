import pygame
import torch
from dataclasses import dataclass

from game_mechanics_display import GAME_DISPLAY
from game_mechanics_nondisplay import GAME_NON_DISPLAY


pygame.init()
pygame.display.set_caption("Snake")

WIDTH, HEIGHT = 900, 900
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
CLOCK = pygame.time.Clock()

GREEN = (19, 145, 57)
LIGHTGREEN = (48, 176, 86)
BLUE = (8, 77, 161)
RED = (161, 77, 8)
BLACK = (0, 0, 0)

FONT = pygame.font.Font("freesansbold.ttf", 32)

# Game Config
START_LENGTH = 4
MAP_SIZE = 30

# NN Config
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
N_ACTIONS = 4

ACTION_OPTIONS = ["up", "right", "down", "left"]
N_EPISODES = 1024

# If GPU available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Gather all constants in one object
@dataclass
class CONSTANTS:
    WIDTH: int
    HEIGHT: int
    SCREEN: pygame.display.set_mode  # Screen object
    CLOCK: pygame.time.Clock

    GREEN: tuple
    LIGHTGREEN: tuple
    BLUE: tuple
    RED: tuple
    BLACK: tuple
    FONT: tuple

    START_LENGTH: int
    MAP_SIZE: int

    BATCH_SIZE: int
    GAMMA: float
    EPS_START: float
    EPS_END: float
    EPS_DECAY: float
    TAU: float
    LR: float

    N_ACTIONS: int
    ACTION_OPTIONS: list
    N_EPISODES: int

    DEVICE: torch.device


SNAKE_GAME_CONSTANTS = CONSTANTS(
    WIDTH,
    HEIGHT,
    SCREEN,
    CLOCK,
    GREEN,
    LIGHTGREEN,
    BLUE,
    RED,
    BLACK,
    FONT,
    START_LENGTH,
    MAP_SIZE,
    BATCH_SIZE,
    GAMMA,
    EPS_START,
    EPS_END,
    EPS_DECAY,
    TAU,
    LR,
    N_ACTIONS,
    ACTION_OPTIONS,
    N_EPISODES,
    DEVICE,
)


# All user defined parameters (later altered by user)
user_defined = {
    "autonomous": False,
    "load_model": None,
    "threshold": False,
}

autonomous = True if input("Autonomous: ").lower() == "y" else False
user_defined["autonomous"] = autonomous
display = True if input("Display: ").lower() == "y" else False

if not autonomous:
    snake_game = GAME_DISPLAY(user_defined, SNAKE_GAME_CONSTANTS)
    snake_game.run_user_game_loop()
    quit()

load_model = input("Load model: ")
load_model = load_model if load_model != "" else None
user_defined["load_model"] = load_model
use_threshold = True if input("Threshold: ").lower() == "y" else False
user_defined["threshold"] = use_threshold
model_name = input("Model name: ")
model_name = model_name if model_name != "" else "Unnamed AI model"
user_defined["model_name"] = model_name

# Only init one time, since policy network is inside
snake_game = (
    GAME_DISPLAY(user_defined, SNAKE_GAME_CONSTANTS)
    if display
    else GAME_NON_DISPLAY(user_defined, SNAKE_GAME_CONSTANTS)
)

for i_episode in range(N_EPISODES):
    snake_game.run_autonomous_game_loop()

# Only quit pygame after the entire training loop is done
pygame.quit()

# TODO: Adjust this info for snake AI
snake_game.snake.handler.save_model(
    snake_game.snake.policy_net.state_dict(),
    {
        "model name": user_defined["model_name"],
        "batch size": BATCH_SIZE,
        "episodes": N_EPISODES,
        "learning rate": LR,
        "complete": True,
        "top score": snake_game.top_score,
        "top SAS": snake_game.small_score_average_max,
    },
)

print("Saved as complete model")
