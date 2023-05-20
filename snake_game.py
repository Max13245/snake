import pygame, sys

pygame.init()

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
        self.start_position = (16, 15)
        self.body = []
        self.initiate_body()

    def create_limb(self):
        pass

    def initiate_body(self):
        for i in range(self.length):
            limb = pygame.Rect((self.start_position[0] - i) * self.size_x, 
                               self.start_position[1] * self.size_y, 
                               self.limb_size_x, self.limb_size_y)
            self.body.append(limb)

    def draw_snake(self):
        for limb in self.body:
            pygame.draw.ellipse(screen, BLUE, limb)

class MAP:
    def __init__(self, size):
        self.x_blocks, self.y_blocks = width / size, height / size
        self.block_size = size
        self.tiles = []
        self.smap = self.create_map()
        self.apple = self.create_apple()
        self.apple_possition = (23, 15)
        self.snake = SNAKE(self.x_blocks, self.y_blocks)
        self.last_direction = "right"

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
                    self.tiles[self.apple_possition[0]][self.apple_possition[1]][0])

    def reposition_apple(self):
        pass

    def create_apple(self):
        apple_img = pygame.image.load("apple.png")
        return pygame.transform.scale(apple_img, 
                                      (int(self.x_blocks), 
                                       int(self.y_blocks)))

    def run_game_loop(self):
        while True:
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
            self.snake.draw_snake()
            pygame.display.update()
            clock.tick(60)

snake_map = MAP(30)
snake_map.run_game_loop()