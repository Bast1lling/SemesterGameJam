import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up the display
screen_width, screen_height = 1920, 1080
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Draw Image Example')
# Load the image
image = pygame.image.load('SemesterGameJam/assets/page_1.png')

# creating a bool value which checks
# if game is running
running = True

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill((0, 0, 0))  # Fill with black color

    # Draw the image at the specified position (x, y)
    x, y = 100, 100
    screen.blit(image, (x, y))

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
