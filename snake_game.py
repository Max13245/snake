import pygame
from random import randint, sample

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import math
from collections import namedtuple, deque
import os


pygame.init()
pygame.display.set_caption("Snake")

width, height = 900, 900
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

GREEN = (19, 145, 57)
LIGHTGREEN = (48, 176, 86)
BLUE = (8, 77, 161)
RED = (161, 77, 8)
BLACK = (0, 0, 0)

font = pygame.font.Font("freesansbold.ttf", 32)

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
        self.layer1 = nn.Linear(9, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return x


class SNAKE_BRAIN:
    def __init__(self, load_model):
        if load_model:
            self.policy_net = DQN().to(device)
            # TODO: Don't use try except (use handler for this anyway)
            try:
                self.policy_net.load_state_dict(
                    torch.load(f"./models/model_{load_model}")
                )
            except:
                self.policy_net.load_state_dict(
                    torch.load(f"./models/model_{load_model}_incomplete")
                )
            self.policy_net.eval()
        else:
            self.policy_net = DQN().to(device)

        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)


class SNAKE_CALCULATE(SNAKE_BRAIN):
    def __init__(self, load_model) -> None:
        super().__init__(load_model)

        self.length = START_LENGTH
        self.position = (16, 15)
        self.body = []
        self.initiate_body()

        self.previous_direction = None
        self.direction = "right"

    def initiate_body(self):
        # Reverse range, so head is at the end of the list
        for i in range(self.length, 0, -1):
            body_position = (self.position[0] - i, self.position[1])
            self.body.append(body_position)

    def correct_reverse(self):
        if self.previous_direction == "up" and self.direction == "down":
            self.direction == "up"
        elif self.previous_direction == "right" and self.direction == "left":
            self.direction == "right"
        elif self.previous_direction == "down" and self.direction == "up":
            self.direction == "down"
        elif self.previous_direction == "left" and self.direction == "right":
            self.direction == "left"

    def move_head(self):
        self.correct_reverse()
        body_x = self.body[-1][0]
        body_y = self.body[-1][1]
        if self.direction == "up":
            self.body.append((body_x, body_y - 1))
        elif self.direction == "right":
            self.body.append((body_x + 1, body_y))
        elif self.direction == "down":
            self.body.append((body_x, body_y + 1))
        elif self.direction == "left":
            self.body.append((body_x - 1, body_y))

    def move(self, apple_overlap):
        # Don't delete head when snake gets apple
        if not apple_overlap:
            del self.body[-1]

    def wall_collision(self):
        body_x = self.body[-1][0]
        body_y = self.body[-1][1]
        if body_x < 0 or body_x > 30:
            return True
        if body_y < 0 or body_y > 30:
            return True
        return False

    def tangled(self):
        head_position = self.body[-1]
        for body_indx in range(1, len(self.body)):
            if self.body[body_indx] == head_position:
                return True
        return False


class GAME_NON_DISPLAY:
    def __init__(self, size, user_defined):
        self.block_size = size

        # This also creates the apple (tuple of coördinates)
        self.reposition_apple()

        self.snake = SNAKE_CALCULATE(user_defined["load_model"])

        self.apple_overlap = False
        self.max_relu_value = 0.0

        # For calculating reward for closing in on apple
        self.maximum_apple_radius_reward = 0.2
        self.previous_apple_distance = self.calculate_apple_distance()

        # Maximum apple reward, will be less based on snakes length
        self.maximum_apple_reward = 1

        # Info for screen
        self.n_episodes = 1
        self.n_steps = 1
        self.n_batches = 1
        self.score = 0
        self.top_score = 0
        self.average_score = 0
        self.average_score_max = 0
        self.average_total_sum = 30
        self.small_score_average_data = []
        self.small_score_average = 0
        self.small_score_average_max = 0
        self.previous_loss = 0
        self.use_threshold = user_defined["threshold"]
        self.random_threshold = round(
            EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * self.n_steps / EPS_DECAY),
            2,
        )

        self.show_information = True

    def reposition_apple(self):
        self.apple_possition_x = randint(0, (self.block_size - 1))
        self.apple_possition_y = randint(0, (self.block_size - 1))

    def is_apple_overlap(self):
        for cords in self.snake.body:
            if cords == (self.apple_possition_x, self.apple_possition_y):
                self.reposition_apple()
                return True
        return False

    def calculate_apple_distance(self):
        x_distance = np.abs(
            self.apple_possition_x - self.snake.body[0].x / self.x_blocks
        )
        x_distance = math.floor(x_distance)
        y_distance = np.abs(
            self.apple_possition_y - self.snake.body[0].y / self.y_blocks
        )
        y_distance = math.floor(y_distance)
        total_distance = x_distance + y_distance
        return total_distance

    def move_snake(self):
        # Move body parts before head
        self.snake.move()
        self.snake.move_head()
        self.snake.previous_direction = self.snake.direction

    def update_information_types(self):
        # Only add threshold to the list if use_threshold is true
        self.information_types = [
            ("Episode", self.n_episodes),
            ("Step", self.n_steps),
            ("Batch", self.n_batches),
            ("Score", self.score),
            ("Top Score", self.top_score),
            ("Average Score", self.average_score),
            ("Average Score Max", self.average_score_max),
            ("Small Average Score", self.small_score_average),
            ("Small Average Score Max", self.small_score_average_max),
            ("Loss", self.previous_loss),
        ] + ([("Threshold", self.random_threshold)] if self.use_threshold else [])

    def show_info(self):  # Change to terminal
        if not self.show_information:
            return

        self.update_information_types()

        text_x_position, text_y_position = 5, 5
        for info in self.information_types:
            text = font.render(f"{info[0]}: {info[1]}", True, BLACK)
            text_rect = text.get_rect()
            text_rect.topleft = (text_x_position, text_y_position)
            screen.blit(text, text_rect)
            text_y_position += text_rect.height + 5

    def get_state(self):
        """
        1. Head position                             2
        2. Apple position                            2
        3. Up and down collision distance            2
        4. Left and Right collistion distance        2
        5. Current direction                         1
        """

        head_position = [self.snake.body[-1][0], self.snake.body[-1][1]]
        apple_position = [self.apple_possition_x, self.apple_possition_y]

        head_x, head_y = self.snake.body[-1]
        left_row = head_x
        right_row = self.block_size - head_x
        above_column = head_y
        beneath_column = self.block_size - head_y

        for body_indx in range(1, len(self.snake.body)):
            if self.snake.body[body_indx][1] == head_y:
                horizontal_distance = self.snake.body[body_indx][0] - head_x
                # Asumes the horizontal distance is not zero
                if horizontal_distance > 0 and horizontal_distance < right_row:
                    right_row = horizontal_distance
                    continue

                if abs(horizontal_distance) < left_row:
                    left_row = abs(horizontal_distance)
            elif self.snake.body[body_indx][0] == head_x:
                vertical_distance = self.snake.body[body_indx][1] - head_y
                # Asumes the vertical distance is not zero
                if vertical_distance > 0 and vertical_distance < beneath_column:
                    beneath_column = vertical_distance
                    continue

                if abs(vertical_distance) < above_column:
                    above_column = abs(vertical_distance)

        distances = [above_column, beneath_column, left_row, right_row]

        current_direction = ACTION_OPTIONS.index(self.last_direction)

        state = head_position + apple_position + distances + [current_direction]
        return np.array(state)

    def check_quit_event(self):  # Change to terminal somehow
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    self.show_information ^= True
        return False

    def get_apple_radius_reward(self):  # GAME_MECH
        # If apple is recieved in this round then skip radius reward
        if self.apple_overlap:
            return 0

        distance = self.calculate_apple_distance()

        # If self.previous_apple_distance is at apple location don't calc reward
        if not self.previous_apple_distance:
            reward = 0
        elif self.previous_apple_distance < distance:
            # Return negative reward
            # Must as big as positive reward, otherwise snake
            # might circle around apple to maximize reward
            reward = -self.maximum_apple_radius_reward
        elif distance < self.previous_apple_distance:
            # Return positive reward
            reward = self.maximum_apple_radius_reward
        else:
            reward = 0

        self.previous_apple_distance = distance
        return reward

    def calculate_scores(self):  # GAME_MECH
        self.score = self.snake.length - START_LENGTH
        if self.score > self.top_score:
            self.top_score = self.score

    def calculate_averages(self):  # GAME_MECH
        self.average_score = round(
            (self.average_score * (self.n_episodes - 1) + self.score) / self.n_episodes,
            2,
        )

        if self.average_score > self.average_score_max:
            self.average_score_max = self.average_score

        # Calculate the score average of the x last episodes
        self.small_score_average_data.append(self.score)
        if len(self.small_score_average_data) >= self.average_total_sum + 1:
            # TODO: Use deque
            self.small_score_average_data = self.small_score_average_data[1:]
        self.small_score_average = round(
            sum(self.small_score_average_data) / self.average_total_sum, 2
        )

        if self.small_score_average > self.small_score_average_max:
            self.small_score_average_max = self.small_score_average

    def game_mechanics(self):  # Change
        self.move_snake(self.apple_overlap)
        self.apple_overlap = self.is_apple_overlap()
        self.calculate_scores()
        self.show_info()

    def store_state_action(self, state, action, next_state, reward):  # Same
        # Create reward tensor
        reward_tensor = torch.tensor([reward], device=device)

        # Create a transition
        self.snake.memory.push(state, action, next_state, reward_tensor)

    def soft_update_target_net(self):  # SNAKE_BRAIN
        # Soft update of the target network's weights
        target_net_state_dict = self.snake.target_net.state_dict()
        policy_net_state_dict = self.snake.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.snake.target_net.load_state_dict(target_net_state_dict)

    def calculate_apple_reward(self) -> float:  # Same
        return math.sqrt(self.maximum_apple_reward - self.snake.length / MAP_SIZE**2)

    def calculate_collision_reward(self) -> float:  # Same
        # TODO Make bigger/smaller?
        return -((self.snake.length / MAP_SIZE**2) ** 2)

    def run_autonomous_game_loop(self):
        state = None
        next_state = None
        truncated = False  # TODO: Necessary?
        while True:
            terminated = False
            if state != None:
                # Transition init data
                observation = self.get_state()
                reward = 0

                # Get positive reward when apple overlap happens
                if self.apple_overlap:
                    # Calculate positive apple reward based on snake length
                    reward = self.calculate_apple_reward()

                if not (reward >= 1):
                    reward += self.get_apple_radius_reward()

                # Get next state
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

                self.store_state_action(state, action, next_state, reward)

            # Get state and format it into a tensor
            if next_state != None:
                state = next_state
            else:
                state = self.get_state()
                state = torch.tensor(
                    state, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Get and perform an action, only when at an intersection
            action = snake_map.select_action(state)
            self.snake.direction = ACTION_OPTIONS[action]

            quit_event = self.check_quit_event()
            if quit_event:
                break

            # Game loop mechanics
            self.game_mechanics()
            pygame.display.update()
            clock.tick(60)

            # Get negative reward when wall_collision or snake gets tangled
            # TODO: Maybe differenciate between wall and snake collision
            if self.snake.wall_collision() or self.snake.tangled():
                observation = None

                reward = self.calculate_collision_reward()

                self.store_state_action(state, action, observation, reward)

                # Adjust average score
                # Must happen before reset map, depends on self.n_episodes
                self.calculate_averages()

                # Use reset map instead of pygame.quit()
                self.reset_map()

                terminated = True

            self.soft_update_target_net()
            # Perform one step of the optimization (on the policy network)
            self.optimize_model()

            # One step done
            self.n_steps += 1

            if terminated or truncated:
                break

        return quit_event

    def select_action(self, state):
        sample = np.random.random()
        step_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.n_steps / EPS_DECAY
        )
        self.random_threshold = round(step_threshold, 2)
        if (
            sample > step_threshold or step_threshold <= 0.06
        ) or not self.use_threshold:
            with torch.no_grad():
                # Will return a list of relu values
                output = self.snake.policy_net(state).max(1)
                if output[0].item() > self.max_relu_value:
                    self.max_relu_value = output[0].item()
                return output.indices.view(1, 1)
        else:
            # Return random values to explore new possibilities
            random_array = np.array(np.random.uniform(0, self.max_relu_value, 4))
            random_action = torch.tensor(random_array).max(0).indices.view(1, 1)
            return random_action

    def optimize_model(self):  # Snake brain?
        if len(self.snake.memory) < BATCH_SIZE:
            return
        transitions = self.snake.memory.sample(BATCH_SIZE)

        # Transpose the batch
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
        self.previous_loss = round(loss.item(), 2)

        # Optimize the model
        self.snake.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.snake.policy_net.parameters(), 100)
        self.snake.optimizer.step()
        self.n_batches += 1

    def reset_map(self):  # Change
        """Reset the map, so pygame.init doesn't have to run every training loop"""
        self.reposition_apple()
        self.last_direction = "right"
        self.direction = "right"
        self.n_episodes += 1

        # Snake values
        self.snake.length = START_LENGTH
        self.snake.position = (16 * self.snake.size_x, 15 * self.snake.size_y)
        self.snake.body = []
        self.snake.initiate_body()

        # Must be called after repositioning snake and apple
        self.previous_apple_distance = self.calculate_apple_distance()


class SNAKE_DISPLAY(SNAKE_BRAIN):
    def __init__(self, size_x, size_y, load_model) -> None:
        super().__init__(load_model)

        # Only create variables needed for current display setting
        self.length = START_LENGTH
        self.speed = int(size_x / 10)
        self.size_x, self.size_y = size_x, size_y
        self.limb_size_x, self.limb_size_y = size_x, size_y
        self.position = (
            16 * self.size_x,
            15 * self.size_y,
        )
        self.current_color = BLUE

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
            pygame.draw.ellipse(screen, self.current_color, limb)

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


class GAME_DISPLAY:
    def __init__(self, size, user_defined):
        self.autonomous = user_defined["autonomous"]
        self.display = user_defined["display"]
        self.x_blocks, self.y_blocks = width / size, height / size
        self.block_size = size
        self.tiles = []
        self.map = self.create_map()
        self.apple = self.create_apple()
        self.reposition_apple()

        if self.display:
            self.snake = SNAKE_DISPLAY(
                self.x_blocks,
                self.y_blocks,
                user_defined["load_model"],
            )
        else:
            self.snake = SNAKE_CALCULATE(user_defined["load_model"])

        self.last_direction = "right"
        self.direction = "right"
        self.apple_overlap = False
        self.max_relu_value = 0.0

        # For calculating reward for closing in on apple
        self.maximum_apple_radius_reward = 0.2
        self.previous_apple_distance = self.calculate_apple_distance()

        # Maximum apple reward, will be less based on snakes length
        self.maximum_apple_reward = 1

        # Info for screen
        self.n_episodes = 1
        self.n_steps = 1
        self.n_batches = 1
        self.score = 0
        self.top_score = 0
        self.average_score = 0
        self.average_score_max = 0
        self.average_total_sum = 30
        self.small_score_average_data = []
        self.small_score_average = 0
        self.small_score_average_max = 0
        self.previous_loss = 0
        self.use_threshold = user_defined["threshold"]
        self.random_threshold = round(
            EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * self.n_steps / EPS_DECAY),
            2,
        )

        self.show_information = True

    def create_map(self):  # MAP
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

    def draw_map(self):  # MAP
        for row in self.tiles:
            for tile in row:
                pygame.draw.rect(screen, tile[1], tile[0])

    def show_apple(self):  # MAP
        screen.blit(
            self.apple, self.tiles[self.apple_possition_x][self.apple_possition_y][0]
        )

    def reposition_apple(self):  # GAME_MECH
        self.apple_possition_x = randint(0, (self.block_size - 1))
        self.apple_possition_y = randint(0, (self.block_size - 1))

    def is_apple_overlap(self):  # GAME_MECH
        if (
            self.apple_possition_x * self.x_blocks == self.snake.body[0].x
            and self.apple_possition_y * self.y_blocks == self.snake.body[0].y
        ):
            self.snake.create_limb()
            self.reposition_apple()
            return True
        return False

    def create_apple(self):  # MAP
        apple_img = pygame.image.load("apple.png")
        return pygame.transform.scale(
            apple_img, (int(self.x_blocks), int(self.y_blocks))
        )

    def calculate_apple_distance(self):  # GAME_MECH
        x_distance = np.abs(
            self.apple_possition_x - self.snake.body[0].x / self.x_blocks
        )
        x_distance = math.floor(x_distance)
        y_distance = np.abs(
            self.apple_possition_y - self.snake.body[0].y / self.y_blocks
        )
        y_distance = math.floor(y_distance)
        total_distance = x_distance + y_distance
        return total_distance

    def at_intersection(self):  # GAME_MECH
        if (
            self.snake.body[0].x % self.x_blocks == 0
            and self.snake.body[0].y % self.y_blocks == 0
        ):
            return True
        return False

    def is_reverse(self, current, new):  # GAME_MECH
        if current == "up" and new == "down":
            return True
        if current == "right" and new == "left":
            return True
        if current == "down" and new == "up":
            return True
        if current == "left" and new == "right":
            return True
        return False

    def move_snake(self):  # GAME_MECH
        if self.at_intersection():
            if not self.is_reverse(self.direction, self.last_direction):
                self.direction = self.last_direction
            self.snake.update_limb_direction()

        # Move head before snake limbs
        self.snake.move_head(self.direction)
        self.snake.move()

    def update_information_types(self):  # GAME_MECH
        if self.autonomous:
            # Only add threshold to the list if use_threshold is true
            self.information_types = [
                ("Episode", self.n_episodes),
                ("Step", self.n_steps),
                ("Batch", self.n_batches),
                ("Score", self.score),
                ("Top Score", self.top_score),
                ("Average Score", self.average_score),
                ("Average Score Max", self.average_score_max),
                ("Small Average Score", self.small_score_average),
                ("Small Average Score Max", self.small_score_average_max),
                ("Loss", self.previous_loss),
            ] + ([("Threshold", self.random_threshold)] if self.use_threshold else [])
        else:
            self.information_types = [
                ("Score", self.score),
            ]

    def show_info(self):  # MAP
        if not self.show_information:
            return

        self.update_information_types()

        text_x_position, text_y_position = 5, 5
        for info in self.information_types:
            text = font.render(f"{info[0]}: {info[1]}", True, BLACK)
            text_rect = text.get_rect()
            text_rect.topleft = (text_x_position, text_y_position)
            screen.blit(text, text_rect)
            text_y_position += text_rect.height + 5

    def user_control(self):  # MAP
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.last_direction = "up"
                elif event.key == pygame.K_RIGHT:
                    self.last_direction = "right"
                elif event.key == pygame.K_DOWN:
                    self.last_direction = "down"
                elif event.key == pygame.K_LEFT:
                    self.last_direction = "left"

        return False

    def run_user_game_loop(self):  # GAME_MECH
        while True:
            truncated = self.user_control()
            if truncated:
                break

            _ = self.game_mechanics()

            if self.snake.wall_collision() or self.snake.tangled():
                pygame.quit()
                break

            pygame.display.update()
            clock.tick(60)

    def get_state(self):
        """
        1. Head position                             2
        2. Apple position                            2
        3. Up and down collision distance            2
        4. Left and Right collistion distance        2
        5. Current direction                         1
        """

        head_position = [
            self.snake.body[0].x / self.x_blocks,
            self.snake.body[0].y / self.y_blocks,
        ]
        apple_position = [self.apple_possition_x, self.apple_possition_y]

        # TODO Funky code, probably inefficient, just testing something
        up_distances = []
        down_distances = []
        left_distances = []
        right_distances = []

        snake_head = self.snake.body[0]
        for limb in self.snake.body[1:]:
            if limb.x == snake_head.x:
                vertical_distance = limb.y - snake_head.y
                if vertical_distance < 0:
                    up_distances.append(vertical_distance * -1)
                elif vertical_distance > 0:
                    down_distances.append(vertical_distance)
            elif limb.y == snake_head.y:
                horizontal_distance = limb.x - snake_head.x
                if horizontal_distance < 0:
                    left_distances.append(horizontal_distance * -1)
                elif horizontal_distance > 0:
                    right_distances.append(horizontal_distance)

        if len(up_distances) == 0:
            up_distances.append(snake_head.y)
        if len(down_distances) == 0:
            down_distances.append(height - snake_head.y)
        if len(left_distances) == 0:
            left_distances.append(snake_head.x)
        if len(right_distances) == 0:
            right_distances.append(width - snake_head.x)

        distances = [
            min(up_distances) / self.y_blocks,
            min(down_distances) / self.y_blocks,
            min(left_distances) / self.x_blocks,
            min(right_distances) / self.x_blocks,
        ]
        current_direction = ACTION_OPTIONS.index(self.last_direction)

        state = head_position + apple_position + distances + [current_direction]
        return np.array(state)

    def AI_control(self, action):  # GAME_MECH
        self.last_direction = ACTION_OPTIONS[action]

    def check_quit_event(self):  # GAME_MECH
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    self.show_information ^= True
        return False

    def get_apple_radius_reward(self):  # GAME_MECH
        # If apple is recieved in this round then skip radius reward
        if self.apple_overlap:
            return 0

        distance = self.calculate_apple_distance()

        # If self.previous_apple_distance is at apple location don't calc reward
        if not self.previous_apple_distance:
            reward = 0
        elif self.previous_apple_distance < distance:
            # Return negative reward
            # Must as big as positive reward, otherwise snake
            # might circle around apple to maximize reward
            reward = -self.maximum_apple_radius_reward
        elif distance < self.previous_apple_distance:
            # Return positive reward
            reward = self.maximum_apple_radius_reward
        else:
            reward = 0

        self.previous_apple_distance = distance
        return reward

    def calculate_scores(self):  # GAME_MECH
        self.score = self.snake.length - START_LENGTH
        if self.score > self.top_score:
            self.top_score = self.score

    def calculate_averages(self):  # GAME_MECH
        self.average_score = round(
            (self.average_score * (self.n_episodes - 1) + self.score) / self.n_episodes,
            2,
        )

        if self.average_score > self.average_score_max:
            self.average_score_max = self.average_score

        # Calculate the score average of the x last episodes
        self.small_score_average_data.append(self.score)
        if len(self.small_score_average_data) >= self.average_total_sum + 1:
            # TODO: Use deque
            self.small_score_average_data = self.small_score_average_data[1:]
        self.small_score_average = round(
            sum(self.small_score_average_data) / self.average_total_sum, 2
        )

        if self.small_score_average > self.small_score_average_max:
            self.small_score_average_max = self.small_score_average

    def game_mechanics(self):  # GAME_MECH
        screen.fill((0, 0, 0))
        self.draw_map()
        self.show_apple()
        self.move_snake()
        self.apple_overlap = self.is_apple_overlap()
        self.snake.draw_snake()
        self.calculate_scores()
        self.show_info()

    def store_state_action(self, state, action, next_state, reward):  # SNAKE_HEAD
        # Create reward tensor
        reward_tensor = torch.tensor([reward], device=device)

        # Create a transition
        self.snake.memory.push(state, action, next_state, reward_tensor)

    def soft_update_target_net(self):  # SNAKE_BRAIN
        # Soft update of the target network's weights
        target_net_state_dict = self.snake.target_net.state_dict()
        policy_net_state_dict = self.snake.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.snake.target_net.load_state_dict(target_net_state_dict)

    def calculate_apple_reward(self) -> float:  # GAME_MECH
        return math.sqrt(self.maximum_apple_reward - self.snake.length / MAP_SIZE**2)

    def calculate_collision_reward(self) -> float:  # GAME_MECH
        # TODO Make bigger/smaller?
        return -((self.snake.length / MAP_SIZE**2) ** 2)

    def run_autonomous_game_loop(self):  # GAME_MECH
        state = None
        next_state = None
        truncated = False  # TODO: Necessary?
        while True:
            if not self.at_intersection():
                # Game loop mechanics
                self.game_mechanics()
                pygame.display.update()
                clock.tick(60)

                continue

            terminated = False
            if state != None:
                # Transition init data
                observation = self.get_state()
                reward = 0

                # Get positive reward when apple overlap happens
                if self.apple_overlap:
                    # Calculate positive apple reward based on snake length
                    reward = self.calculate_apple_reward()

                if not (reward >= 1):
                    reward += self.get_apple_radius_reward()

                # Get next state
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

                self.store_state_action(state, action, next_state, reward)

            # Get state and format it into a tensor
            if next_state != None:
                state = next_state
            else:
                state = self.get_state()
                state = torch.tensor(
                    state, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Get and perform an action, only when at an intersection
            action = snake_map.select_action(state)
            self.AI_control(action)

            quit_event = self.check_quit_event()
            if quit_event:
                break

            # Game loop mechanics
            self.game_mechanics()
            pygame.display.update()
            clock.tick(60)

            # Get negative reward when wall_collision or snake gets tangled
            # TODO: Maybe differenciate between wall and snake collision
            if self.snake.wall_collision() or self.snake.tangled():
                observation = None

                reward = self.calculate_collision_reward()

                self.store_state_action(state, action, observation, reward)

                # Adjust average score
                # Must happen before reset map, depends on self.n_episodes
                self.calculate_averages()

                # Use reset map instead of pygame.quit()
                self.reset_map()

                terminated = True

            self.soft_update_target_net()
            # Perform one step of the optimization (on the policy network)
            self.optimize_model()

            # One step done
            self.n_steps += 1

            if terminated or truncated:
                break

        return quit_event

    def select_action(self, state):  # ?
        sample = np.random.random()
        step_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.n_steps / EPS_DECAY
        )
        self.random_threshold = round(step_threshold, 2)
        if (
            sample > step_threshold or step_threshold <= 0.06
        ) or not self.use_threshold:
            with torch.no_grad():
                self.snake.current_color = BLUE
                # Will return a list of relu values
                output = self.snake.policy_net(state).max(1)
                if output[0].item() > self.max_relu_value:
                    self.max_relu_value = output[0].item()
                return output.indices.view(1, 1)
        else:
            self.snake.current_color = RED
            # Return random values to explore new possibilities
            random_array = np.array(np.random.uniform(0, self.max_relu_value, 4))
            random_action = torch.tensor(random_array).max(0).indices.view(1, 1)
            return random_action

    def optimize_model(self):  # Snake brain?
        if len(self.snake.memory) < BATCH_SIZE:
            return
        transitions = self.snake.memory.sample(BATCH_SIZE)

        # Transpose the batch
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
        self.previous_loss = round(loss.item(), 2)

        # Optimize the model
        self.snake.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.snake.policy_net.parameters(), 100)
        self.snake.optimizer.step()
        self.n_batches += 1

    def reset_map(self):  # GAME_MECH
        """Reset the map, so pygame.init doesn't have to run every training loop"""
        self.reposition_apple()
        self.last_direction = "right"
        self.direction = "right"
        self.n_episodes += 1

        # Snake values
        self.snake.length = START_LENGTH
        self.snake.position = (16 * self.snake.size_x, 15 * self.snake.size_y)
        self.snake.body = []
        self.snake.initiate_body()

        # Must be called after repositioning snake and apple
        self.previous_apple_distance = self.calculate_apple_distance()


def get_n_models(path):
    n_models = len(
        [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
    )
    return n_models


# All user defined parameters (later altered by user)
user_defined = {
    "autonomous": False,
    "display": True,
    "load_model": None,
    "threshold": False,
}


autonomous = True if input("Autonomous: ").lower() == "y" else False
user_defined["autonomous"] = autonomous

display = True if input("Display: ").lower() == "y" else False
user_defined["display"] = display

if not autonomous:
    snake_map = GAME_DISPLAY(MAP_SIZE, user_defined)
    snake_map.run_user_game_loop()
    quit()

load_model = input("Load model: ")
user_defined["load_model"] = load_model
use_threshold = True if input("Threshold: ").lower() == "y" else False
user_defined["threshold"] = use_threshold


# Only init one time, since policy network is inside
snake_map = (
    GAME_DISPLAY(MAP_SIZE, user_defined)
    if display
    else GAME_NON_DISPLAY(MAP_SIZE, user_defined)
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
