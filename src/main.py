# import pygame package
import pygame

# initializing imported module
pygame.init()

# displaying a window of height
# 500 and width 400
pygame.display.set_mode((400, 500))

# Setting name for window
pygame.display.set_caption('GeeksforGeeks')

# creating a bool value which checks
# if game is running
running = True

# Game loop
# keep game running till running is true
while running:

    # Check for event if user has pushed
    # any event in queue
    for event in pygame.event.get():

        # if event is of type quit then set
        # running bool to false
        if event.type == pygame.QUIT:
            running = False
