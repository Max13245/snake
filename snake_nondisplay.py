from snake_brain import SNAKE_BRAIN


class SNAKE_CALCULATE(SNAKE_BRAIN):
    def __init__(self, load_model, constants) -> None:
        super().__init__(load_model, constants)

        self.length = constants.START_LENGTH
        self.position = (16, 15)
        self.body = []
        self.initiate_body()

        self.previous_direction = "right"
        self.direction = "right"

    def initiate_body(self):
        # Reverse range, so head is at the end of the list
        for i in range(self.length, 0, -1):
            body_position = (self.position[0] - i, self.position[1])
            self.body.append(body_position)

    def correct_reverse(self):
        if self.previous_direction == "up" and self.direction == "down":
            self.direction = "up"
        elif self.previous_direction == "right" and self.direction == "left":
            self.direction = "right"
        elif self.previous_direction == "down" and self.direction == "up":
            self.direction = "down"
        elif self.previous_direction == "left" and self.direction == "right":
            self.direction = "left"

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
            del self.body[0]
        else:
            self.length += 1

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
        for body_indx in range(0, len(self.body) - 1):
            if self.body[body_indx] == head_position:
                return True
        return False
