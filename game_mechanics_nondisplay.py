from random import randint
import torch
import torch.nn as nn
import numpy as np
import math

from snake_nondisplay import SNAKE_CALCULATE


class GAME_NON_DISPLAY:
    def __init__(self, user_defined, constants):
        self.constants = constants
        self.block_size = constants.MAP_SIZE

        # This also creates the apple (tuple of coördinates)
        self.reposition_apple()

        self.snake = SNAKE_CALCULATE(
            user_defined["load_model"],
            self.constants.N_ACTIONS,
            self.constants.LR,
            self.constants.DEVICE,
            self.constants.START_LENGTH,
        )

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
        x_distance = np.abs(self.apple_possition_x - self.snake.body[-1][0])
        y_distance = np.abs(self.apple_possition_y - self.snake.body[-1][1])
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

    def show_info(self):
        self.update_information_types()
        print("-" * 50)
        for info in self.information_types:
            print(f"{info[0]}{' ' * (30 - len(info[0]))} =    {info[1]}")
        print("-" * 50)

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

        current_direction = self.constants.ACTION_OPTIONS.index(self.last_direction)

        state = head_position + apple_position + distances + [current_direction]
        return np.array(state)

    def check_quit_event(self):  # Change to terminal somehow
        # Check for quit event
        """for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
        return False"""
        pass

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

    def game_mechanics(self):  # Change
        self.move_snake(self.apple_overlap)
        self.apple_overlap = self.is_apple_overlap()
        self.calculate_scores()
        self.show_info()

    def store_state_action(self, state, action, next_state, reward):  # Same
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

    def calculate_apple_reward(self) -> float:  # Same
        return math.sqrt(
            self.maximum_apple_reward - self.snake.length / self.constants.MAP_SIZE**2
        )

    def calculate_collision_reward(self) -> float:  # Same
        # TODO Make bigger/smaller?
        return -((self.snake.length / self.constants.MAP_SIZE**2) ** 2)

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
            self.snake.direction = self.constants.ACTION_OPTIONS[action]

            quit_event = self.check_quit_event()
            if quit_event:
                break

            # Game loop mechanics
            self.game_mechanics()

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
        step_threshold = self.constants.EPS_END + (
            self.constants.EPS_START - self.constants.EPS_END
        ) * math.exp(-1.0 * self.n_steps / self.constants.EPS_DECAY)
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

    def reset_map(self):  # Change
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
