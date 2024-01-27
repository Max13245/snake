import pygame
import torch
import os
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


def get_n_models(path):
    n_models = len(
        [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
    )
    return n_models


# All user defined parameters (later altered by user)
user_defined = {
    "autonomous": False,
    "load_model": None,
    "threshold": False,
}


autonomous = True if input("Autonomous: ").lower() == "y" else False
user_defined["autonomous"] = autonomous

display = True if input("Display: ").lower() == "y" else False
user_defined["display"] = display

if not autonomous:
    snake_map = GAME_DISPLAY(user_defined, SNAKE_GAME_CONSTANTS)
    snake_map.run_user_game_loop()
    quit()

load_model = input("Load model: ")
user_defined["load_model"] = load_model
use_threshold = True if input("Threshold: ").lower() == "y" else False
user_defined["threshold"] = use_threshold


# Only init one time, since policy network is inside
snake_map = (
    GAME_DISPLAY(user_defined, SNAKE_GAME_CONSTANTS)
    if display
    else GAME_NON_DISPLAY(user_defined, SNAKE_GAME_CONSTANTS)
)

for i_episode in range(N_EPISODES):
    # Initialize the environment and get it's state
    # TODO: Maybe find something else for quit_event variable chain
    quit_event = snake_map.run_autonomous_game_loop()

    # Don't run again in case of quit_event
    if quit_event:
        break

# Only quit pygame after the entire training loop is done
pygame.quit()

# Save model TODO: Use handler (handler need swap model function)
models_path = "./models/"
n_models = get_n_models(models_path)

# Add suffix when model has not completed training
if quit_event:
    print("Saved as incomplete model")
    model_suffix = "_incomplete"
else:
    print("Saved as complete model")
    model_suffix = ""

torch.save(
    snake_map.snake.policy_net.state_dict(),
    f"{models_path}model_{n_models}{model_suffix}",
)
