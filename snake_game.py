import pygame, sys
from random import randint

pygame.init()
pygame.display.set_caption("Snake")

width, height = 900, 900
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

GREEN = (19, 145, 57)
LIGHTGREEN = (48, 176, 86)
BLUE = (8, 77, 161)

class SNAKE:
    def __init__(self, size_x, size_y):
        self.speed = int(size_x / 10)
        self.length = 4
        self.size_x, self.size_y = size_x, size_y
        self.limb_size_x, self.limb_size_y = size_x, size_y
        self.position = (16 * self.size_x, 15 * self.size_y)
        self.body = []
        self.initiate_body()

    def create_limb(self):
        limb = pygame.Rect(self.body[-1].x, self.body[-1].y, self.limb_size_x, self.limb_size_y)
        self.body.append(limb)
        self.length += 1

    def initiate_body(self):
        for i in range(self.length):
            limb = pygame.Rect((self.position[0] - i * self.size_x), 
                               self.position[1], 
                               self.limb_size_x, self.limb_size_y)
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
            x_difference = self.body[i -1].x - self.body[i].x
            x_change = self.find_direction(x_difference)

            y_difference = self.body[i -1].y - self.body[i].y
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
    
    def tangled(self, direction):
        if direction == "up":
            for i in range(1, self.length): 
                if self.body[0].x == self.body[i].x and self.body[0].y - self.limb_size_y == self.body[i].y: 
                    return True
        elif direction == "right":
            for i in range(1, self.length):
                if self.body[0].x + self.limb_size_x == self.body[i].x and self.body[0].y == self.body[i].y:
                    return True
        elif direction == "down":
            for i in range(1, self.length):
                if self.body[0].x == self.body[i].x and self.body[0].y + self.limb_size_y == self.body[i].y:
                    return True
        elif direction == "left":
            for i in range(1, self.length):
                if self.body[0].x - self.limb_size_x == self.body[i].x and self.body[0].y == self.body[i].y:
                    return True
        return False

class MAP:
    def __init__(self, size):
        self.x_blocks, self.y_blocks = width / size, height / size
        self.block_size = size
        self.tiles = []
        self.smap = self.create_map()
        self.apple = self.create_apple()
        self.apple_possition_x, self.apple_possition_y = 23, 15
        self.snake = SNAKE(self.x_blocks, self.y_blocks)
        self.last_direction = "right"
        self.direction = "right"

    def create_map(self):
        for i in range(self.block_size):
            row = []
            for j in range(self.block_size):
                tile = pygame.Rect(i * self.block_size, j * self.block_size, 
                                   self.x_blocks, self.y_blocks)
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
        screen.blit(self.apple, 
                    self.tiles[self.apple_possition_x][self.apple_possition_y][0])

    def reposition_apple(self):
        self.apple_possition_x = randint(0, (self.block_size - 1))
        self.apple_possition_y = randint(0, (self.block_size - 1))

    def apple_overlap(self):
        if self.apple_possition_x * self.x_blocks == self.snake.body[0].x and self.apple_possition_y * self.y_blocks == self.snake.body[0].y:
            self.snake.create_limb()
            self.reposition_apple()

    def create_apple(self):
        apple_img = pygame.image.load("apple.png")
        return pygame.transform.scale(apple_img, 
                                      (int(self.x_blocks), 
                                       int(self.y_blocks)))
    
    def at_intersection(self):
        if self.snake.body[0].x % self.x_blocks == 0 and self.snake.body[0].y % self.y_blocks == 0:
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

    def run_game_loop(self):
        while(True):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.last_direction = "up"
                    elif event.key == pygame.K_RIGHT:
                        self.last_direction = "right"
                    elif event.key == pygame.K_DOWN:
                        self.last_direction = "down"
                    elif event.key == pygame.K_LEFT:
                        self.last_direction = "left"

            screen.fill((0, 0, 0))
            self.draw_map()
            self.show_apple()
            self.move_snake()
            self.apple_overlap()
            self.snake.draw_snake()

            if self.snake.wall_collision() or self.snake.tangled(self.direction):
                pygame.quit()
                sys.exit()
                
            pygame.display.update()
            clock.tick(60)

snake_map = MAP(30)
snake_map.run_game_loop()