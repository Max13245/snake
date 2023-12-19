import pygame, sys
from random import randint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import math

pygame.init()
pygame.display.set_caption("Snake")

width, height = 900, 900
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

GREEN = (19, 145, 57)
LIGHTGREEN = (48, 176, 86)
BLUE = (8, 77, 161)
BLACK = (0, 0, 0)

font = pygame.font.Font("freesansbold.ttf", 32)

# Game Config
start_length = 4
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

# If GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(MAP_SIZE ^ 2 + 1, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.sigmoid(self.layer3(x))
        return x


class SNAKE:
    def __init__(self, size_x, size_y, autonomous):
        if autonomous:
            """
            NN inputs:
            1. Current direction
                1 = Up
                2 = Right
                3 = Down
                4 = Left
            2. Every square has a value
                1 = Nothing on square
                2 = Apple on square
                3 = Snake body without turn on square
                4 = Snake body with turn on square
                5 = Snake head on square
            """
            self.policy_net = DQN().to(device)
            self.target_net = DQN().to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())

            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)

        self.speed = int(size_x / 10)
        self.length = start_length
        self.size_x, self.size_y = size_x, size_y
        self.limb_size_x, self.limb_size_y = size_x, size_y
        self.position = (16 * self.size_x, 15 * self.size_y)
        self.body = []
        self.initiate_body()

    def create_limb(self):
        limb = pygame.Rect(
            self.body[-1].x, self.body[-1].y, self.limb_size_x, self.limb_size_y
        )
        self.body.append(limb)
        self.length += 1

    def initiate_body(self):
        for i in range(self.length):
            limb = pygame.Rect(
                (self.position[0] - i * self.size_x),
                self.position[1],
                self.limb_size_x,
                self.limb_size_y,
            )
            self.body.append(limb)

    def draw_snake(self):
        for limb in self.body:
            pygame.draw.ellipse(screen, BLUE, limb)

    def move_head(self, direction):
        if direction == "up":
            self.body[0].move_ip(0, -self.speed)
        elif direction == "right":
            self.body[0].move_ip(self.speed, 0)
        elif direction == "down":
            self.body[0].move_ip(0, self.speed)
        elif direction == "left":
            self.body[0].move_ip(-self.speed, 0)

    def find_direction(self, difference):
        if difference == 0:
            return 0
        elif difference < 0:
            return -self.speed
        return self.speed

    def update_limb_direction(self):
        self.moves = []
        # Loop through all the body parts except the head
        for i in range(1, self.length):
            x_difference = self.body[i - 1].x - self.body[i].x
            x_change = self.find_direction(x_difference)

            y_difference = self.body[i - 1].y - self.body[i].y
            y_change = self.find_direction(y_difference)

            self.moves.append((x_change, y_change))

    def move(self):
        for j in range(1, self.length):
            self.body[j].move_ip(self.moves[j - 1][0], self.moves[j - 1][1])

    def wall_collision(self):
        if self.body[0].x < 0 or self.body[0].x + self.limb_size_x > width:
            return True
        elif self.body[0].y < 0 or self.body[0].y + self.limb_size_y > height:
            return True
        return False

    def tangled(self):
        for i in range(3, self.length):
            if (
                self.body[i].y < self.body[0].y < self.body[i].y + self.limb_size_y
            ) or (
                self.body[i].y
                < self.body[0].y + self.limb_size_y
                < self.body[i].y + self.limb_size_y
            ):
                if self.body[i].x < self.body[0].x < self.body[i].x + self.limb_size_x:
                    return True
                if (
                    self.body[i].x
                    < self.body[0].x + self.limb_size_x
                    < self.body[i].x + self.limb_size_x
                ):
                    return True


class MAP:
    def __init__(self, size, autonomous):
        self.autonomous = autonomous
        self.x_blocks, self.y_blocks = width / size, height / size
        self.block_size = size
        self.tiles = []
        self.smap = self.create_map()
        self.apple = self.create_apple()
        self.apple_possition_x, self.apple_possition_y = 23, 15
        self.snake = SNAKE(self.x_blocks, self.y_blocks, self.autonomous)
        self.last_direction = "right"
        self.direction = "right"

    def create_map(self):
        for i in range(self.block_size):
            row = []
            for j in range(self.block_size):
                tile = pygame.Rect(
                    i * self.block_size,
                    j * self.block_size,
                    self.x_blocks,
                    self.y_blocks,
                )
                if i % 2 == 0:
                    if j % 2 == 0:
                        row.append((tile, LIGHTGREEN))
                    else:
                        row.append((tile, GREEN))
                else:
                    if j % 2 == 0:
                        row.append((tile, GREEN))
                    else:
                        row.append((tile, LIGHTGREEN))

            self.tiles.append(row)

    def draw_map(self):
        for row in self.tiles:
            for tile in row:
                pygame.draw.rect(screen, tile[1], tile[0])

    def show_apple(self):
        screen.blit(
            self.apple, self.tiles[self.apple_possition_x][self.apple_possition_y][0]
        )

    def reposition_apple(self):
        self.apple_possition_x = randint(0, (self.block_size - 1))
        self.apple_possition_y = randint(0, (self.block_size - 1))

    def apple_overlap(self):
        if (
            self.apple_possition_x * self.x_blocks == self.snake.body[0].x
            and self.apple_possition_y * self.y_blocks == self.snake.body[0].y
        ):
            self.snake.create_limb()
            self.reposition_apple()

    def create_apple(self):
        apple_img = pygame.image.load("apple.png")
        return pygame.transform.scale(
            apple_img, (int(self.x_blocks), int(self.y_blocks))
        )

    def at_intersection(self):
        if (
            self.snake.body[0].x % self.x_blocks == 0
            and self.snake.body[0].y % self.y_blocks == 0
        ):
            return True
        return False

    def is_reverse(self, current, new):
        if current == "up" and new == "down":
            return True
        if current == "right" and new == "left":
            return True
        if current == "down" and new == "up":
            return True
        if current == "left" and new == "right":
            return True
        return False

    def move_snake(self):
        if self.at_intersection():
            if not self.is_reverse(self.direction, self.last_direction):
                self.direction = self.last_direction
            self.snake.update_limb_direction()

        # Move body parts before head
        self.snake.move()
        self.snake.move_head(self.direction)

    def show_score(self):
        text = font.render(f"Score: {self.snake.length - start_length}", True, BLACK)
        textRect = text.get_rect()
        screen.blit(text, textRect)

    def user_control(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.last_direction = "up"
                elif event.key == pygame.K_RIGHT:
                    self.last_direction = "right"
                elif event.key == pygame.K_DOWN:
                    self.last_direction = "down"
                elif event.key == pygame.K_LEFT:
                    self.last_direction = "left"

    def AI_control(self):
        pass

    def run_game_loop(self):
        if self.autonomous.lower() == "y":
            move_function = self.AI_control
        else:
            move_function = self.user_control

        while True:
            move_function()

            screen.fill((0, 0, 0))
            self.draw_map()
            self.show_apple()
            self.move_snake()
            self.apple_overlap()
            self.snake.draw_snake()
            self.show_score()

            if self.snake.wall_collision() or self.snake.tangled():
                pygame.quit()
                sys.exit()

            pygame.display.update()
            clock.tick(60)


steps_done = 0


def select_action(state, policy_net):
    sample = np.random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # Will return a list of sigmoid values
            return policy_net(state)
    else:
        # Return random values to explore new possibilities
        return np.array(np.random.random_sample(4))


autonomous = input("Autonomous: ")
snake_map = MAP(MAP_SIZE, autonomous)
snake_map.run_game_loop(autonomous)
