import pygame
from random import randint, sample

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import math
from collections import namedtuple, deque


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

ACTION_OPTIONS = ["up", "right", "down", "left"]
N_EPISODES = 128

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# If GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(MAP_SIZE * MAP_SIZE + 1, 128)
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
                2 = Snake body on square
                3 = Snake head on square
                4 = Apple on square
            """
            self.policy_net = DQN().to(device)
            self.target_net = DQN().to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())

            self.optimizer = optim.AdamW(
                self.policy_net.parameters(), lr=LR, amsgrad=True
            )
            self.memory = ReplayMemory(10000)

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
        self.map = self.create_map()
        self.apple = self.create_apple()
        self.apple_possition_x, self.apple_possition_y = 23, 15
        self.snake = SNAKE(self.x_blocks, self.y_blocks, self.autonomous)
        self.last_direction = "right"
        self.direction = "right"
        self.n_episodes = 1

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
            return True
        return False

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

    def run_user_game_loop(self):
        while True:
            self.user_control()

            screen.fill((0, 0, 0))
            self.draw_map()
            self.show_apple()
            self.move_snake()
            self.apple_overlap()
            self.snake.draw_snake()
            self.show_score()

            if self.snake.wall_collision() or self.snake.tangled():
                pygame.quit()

            pygame.display.update()
            clock.tick(60)

    def get_state(self):
        flat_map_values = np.full(shape=self.block_size * self.block_size, fill_value=1)

        # Set all body values on flat map to 2
        for limb_indx in range(1, len(self.snake.body)):
            limb_location = int(
                self.snake.body[limb_indx].x / self.x_blocks
                + (self.snake.body[limb_indx].y / self.y_blocks - 1) * self.block_size
            )
            flat_map_values[limb_location] = 2

        # Set head value to 3 on flat map
        head_location = int(
            self.snake.body[0].x / self.x_blocks
            + (self.snake.body[0].y / self.y_blocks - 1) * self.block_size
        )
        flat_map_values[head_location] = 3

        # Set apple value to 4 on flat map
        apple_location = int(
            self.apple_possition_x + (self.apple_possition_y - 1) * self.block_size
        )
        flat_map_values[apple_location] = 4

        # Append current direction (as a number from 1 to 4)
        state = np.append(
            flat_map_values, ACTION_OPTIONS.index(self.last_direction) + 1
        )

        return state

    def AI_control(self, action):
        direction_index = np.argmax(action)
        self.direction = ACTION_OPTIONS[direction_index]

    def run_autonomous_game_loop(self):
        while True:
            # Get state and format it into a tensor
            state = self.get_state()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            # Get and perform an action
            action = snake_map.select_action(state)
            self.AI_control(action)

            # Game loop mechanics
            screen.fill((0, 0, 0))
            self.draw_map()
            self.show_apple()
            self.move_snake()
            apple_overlap = self.apple_overlap()
            self.snake.draw_snake()
            self.show_score()

            # Transition init data
            observation = self.get_state()
            reward = 0
            terminated = False
            truncated = False

            # Get positive reward when apple overlap happens
            if apple_overlap:
                reward = 10

            # Get negative reward when wall_collision or snake gets tangled
            # TODO: Maybe differenciate between wall and snake collision
            if self.snake.wall_collision() or self.snake.tangled():
                # Use reset map instead of pygame.quit()
                self.reset_map()

                observation = None
                reward = -10
                terminated = True

            # Game loop mechanics
            pygame.display.update()
            clock.tick(60)

            # Create reward tensor
            reward = torch.tensor([reward], device=device)

            # Get next state
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Create a transition
            self.snake.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Soft update of the target network's weights
            target_net_state_dict = self.snake.target_net.state_dict()
            policy_net_state_dict = self.snake.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            self.snake.target_net.load_state_dict(target_net_state_dict)

            if terminated or truncated:
                break

        # Perform one step of the optimization (on the policy network)
        self.optimize_model()

    def select_action(self, state):
        sample = np.random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.n_episodes / EPS_DECAY
        )
        self.n_episodes += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # Will return a list of sigmoid values
                return self.snake.policy_net(state)
        else:
            # Return random values to explore new possibilities
            return np.array(np.random.random_sample(4))

    def optimize_model(self):
        transitions = self.snake.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.snake.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.snake.target_net(non_final_next_states).max(1).values
            )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.snake.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.snake.policy_net.parameters(), 100)
        self.snake.optimizer.step()

    def reset_map(self):
        """Reset the map, so pygame.init doesn't have to run every training loop"""
        self.tiles = []
        self.apple_possition_x, self.apple_possition_y = 23, 15
        self.last_direction = "right"
        self.direction = "right"
        self.n_episodes += 1


# TODO: Temporary static statement
# autonomous = input("Autonomous: ")
autonomous = True
if not autonomous:
    snake_map = MAP(MAP_SIZE, autonomous)
    snake_map.run_user_game_loop(autonomous)
else:
    # Only init one time, since policy network is inside
    snake_map = MAP(MAP_SIZE, autonomous)
    for i_episode in range(N_EPISODES):
        # Initialize the environment and get it's state
        snake_map.run_autonomous_game_loop()
