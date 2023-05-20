import pygame, sys

# pygame.draw.rect(surface, color, pygame.Rect(30, 30, 60, 60))

pygame.init()

width, height = 900, 900
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

GREEN = (19, 145, 57)
LIGHTGREEN = (48, 176, 86)

class MAP:
    def __init__(self, size):
        self.x_blocks, self.y_blocks = width / size, height / size
        self.block_size = size
        self.tiles = []
        self.smap = self.create_map()

    def create_map(self):
        for i in range(self.block_size):
            for j in range(self.block_size):
                tile = pygame.Rect(i * self.block_size, j * self.block_size, self.x_blocks, self.y_blocks)
                if i % 2 == 0:
                    if j % 2 == 0:
                        color = LIGHTGREEN
                    else:
                        color = GREEN
                else:
                    if j % 2 == 0:
                        color = GREEN
                    else:
                        color = LIGHTGREEN
                self.tiles.append((tile, color))

    def draw_map(self):
        for tile in self.tiles:
            pygame.draw.rect(screen, tile[1], tile[0])

    def run_game_loop(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            screen.fill((0, 0, 0))
            self.draw_map()
            pygame.display.update()
            clock.tick(60)

class SNAKE:
    def __init__(self):
        pass

snake_map = MAP(30)
snake_map.run_game_loop()