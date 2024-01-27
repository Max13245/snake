import torch
import torch.nn as nn
import numpy as np
import math
import pygame
from random import randint

from snake_display import SNAKE_DISPLAY


class GAME_DISPLAY:
    def __init__(self, user_defined, constants):
        self.constants = constants
        self.autonomous = user_defined["autonomous"]
        self.display = user_defined["display"]
        self.x_blocks, self.y_blocks = (
            constants.WIDTH / constants.size,
            constants.HEIGHT / constants.size,
        )
        self.block_size = constants.MAP_SIZE
        self.tiles = []
        self.map = self.create_map()
        self.apple = self.create_apple()
        self.reposition_apple()

        self.snake = SNAKE_DISPLAY(
            self.x_blocks, self.y_blocks, user_defined["load_model"], self.constants
        )

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
            self.constants.EPS_END
            + (self.constants.EPS_START - self.constants.EPS_END)
            * math.exp(-1.0 * self.n_steps / self.constants.EPS_DECAY),
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
                        row.append((tile, self.constants.LIGHTGREEN))
                    else:
                        row.append((tile, self.constants.GREEN))
                else:
                    if j % 2 == 0:
                        row.append((tile, self.constants.GREEN))
                    else:
                        row.append((tile, self.constants.LIGHTGREEN))

            self.tiles.append(row)

    def draw_map(self):  # MAP
        for row in self.tiles:
            for tile in row:
                pygame.draw.rect(self.constants.SCREEN, tile[1], tile[0])

    def show_apple(self):  # MAP
        self.constants.SCREEN.blit(
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

        # Move head before snake limbs
        self.snake.move_head(self.direction)
        self.snake.move()

    def update_information_types(self):
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
            text = self.constants.FONT.render(
                f"{info[0]}: {info[1]}", True, self.constants.BLACK
            )
            text_rect = text.get_rect()
            text_rect.topleft = (text_x_position, text_y_position)
            self.constants.SCREEN.blit(text, text_rect)
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
            self.constants.CLOCK.tick(60)

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
            down_distances.append(self.constants.HEIGHT - snake_head.y)
        if len(left_distances) == 0:
            left_distances.append(snake_head.x)
        if len(right_distances) == 0:
            right_distances.append(self.constants.WIDTH - snake_head.x)

        distances = [
            min(up_distances) / self.y_blocks,
            min(down_distances) / self.y_blocks,
            min(left_distances) / self.x_blocks,
            min(right_distances) / self.x_blocks,
        ]
        current_direction = self.constants.ACTION_OPTIONS.index(self.last_direction)

        state = head_position + apple_position + distances + [current_direction]
        return np.array(state)

    def AI_control(self, action):
        self.last_direction = self.constants.ACTION_OPTIONS[action]

    def check_quit_event(self):
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    self.show_information ^= True
        return False

    def get_apple_radius_reward(self):
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
        self.score = self.snake.length - self.constants.START_LENGTH
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
        self.constants.SCREEN.fill((0, 0, 0))
        self.draw_map()
        self.show_apple()
        self.move_snake()
        self.apple_overlap = self.is_apple_overlap()
        self.snake.draw_snake()
        self.calculate_scores()
        self.show_info()

    def store_state_action(self, state, action, next_state, reward):  # SNAKE_HEAD
        # Create reward tensor
        reward_tensor = torch.tensor([reward], device=self.constants.DEVICE)

        # Create a transition
        self.snake.memory.push(state, action, next_state, reward_tensor)

    def soft_update_target_net(self):  # SNAKE_BRAIN
        # Soft update of the target network's weights
        target_net_state_dict = self.snake.target_net.state_dict()
        policy_net_state_dict = self.snake.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.constants.TAU + target_net_state_dict[key] * (
                1 - self.constants.TAU
            )
        self.snake.target_net.load_state_dict(target_net_state_dict)

    def calculate_apple_reward(self) -> float:  # GAME_MECH
        return math.sqrt(
            self.maximum_apple_reward - self.snake.length / self.constants.MAP_SIZE**2
        )

    def calculate_collision_reward(self) -> float:  # GAME_MECH
        # TODO Make bigger/smaller?
        return -((self.snake.length / self.constants.MAP_SIZE**2) ** 2)

    def run_autonomous_game_loop(self):  # GAME_MECH
        state = None
        next_state = None
        truncated = False  # TODO: Necessary?
        while True:
            if not self.at_intersection():
                # Game loop mechanics
                self.game_mechanics()
                pygame.display.update()
                self.constants.CLOCK.tick(60)

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
                    observation, dtype=torch.float32, device=self.constants.DEVICE
                ).unsqueeze(0)

                self.store_state_action(state, action, next_state, reward)

            # Get state and format it into a tensor
            if next_state != None:
                state = next_state
            else:
                state = self.get_state()
                state = torch.tensor(
                    state, dtype=torch.float32, device=self.constants.DEVICE
                ).unsqueeze(0)

            # Get and perform an action, only when at an intersection
            action = self.select_action(state)
            self.AI_control(action)

            quit_event = self.check_quit_event()
            if quit_event:
                break

            # Game loop mechanics
            self.game_mechanics()
            pygame.display.update()
            self.constants.CLOCK.tick(60)

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
        step_threshold = self.constants.EPS_END + (
            self.constants.EPS_START - self.constants.EPS_END
        ) * math.exp(-1.0 * self.n_steps / self.constants.EPS_DECAY)
        self.random_threshold = round(step_threshold, 2)
        if (
            sample > step_threshold or step_threshold <= 0.06
        ) or not self.use_threshold:
            with torch.no_grad():
                self.snake.current_color = self.constants.BLUE
                # Will return a list of relu values
                output = self.snake.policy_net(state).max(1)
                if output[0].item() > self.max_relu_value:
                    self.max_relu_value = output[0].item()
                return output.indices.view(1, 1)
        else:
            self.snake.current_color = self.constants.RED
            # Return random values to explore new possibilities
            random_array = np.array(np.random.uniform(0, self.max_relu_value, 4))
            random_action = torch.tensor(random_array).max(0).indices.view(1, 1)
            return random_action

    def optimize_model(self):  # Snake brain?
        if len(self.snake.memory) < self.constants.BATCH_SIZE:
            return
        transitions = self.snake.memory.sample(self.constants.BATCH_SIZE)

        # Transpose the batch
        batch = self.snake.memory.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.constants.DEVICE,
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
        next_state_values = torch.zeros(
            self.constants.BATCH_SIZE, device=self.constants.DEVICE
        )
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.snake.target_net(non_final_next_states).max(1).values
            )
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.constants.GAMMA
        ) + reward_batch

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
        self.snake.length = self.constants.START_LENGTH
        self.snake.position = (16 * self.snake.size_x, 15 * self.snake.size_y)
        self.snake.body = []
        self.snake.initiate_body()

        # Must be called after repositioning snake and apple
        self.previous_apple_distance = self.calculate_apple_distance()