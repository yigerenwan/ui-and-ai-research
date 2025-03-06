import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
screen_width = 288
screen_height = 512
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Flappy Bird Clone")

# Colors
background_color = (135, 206, 235)  # Light blue
bird_color = (255, 255, 0)          # Yellow
pipe_color = (0, 128, 0)            # Green

# Game variables
bird_x = 100
bird_y = screen_height // 2
bird_width = 34
bird_height = 24
velocity = 0
gravity = 0.25

pipe_gap = 150
pipe_velocity = -3
pipe_width = 52

pipes = []
pipe_timer = 0
pipe_add_interval = 60  # Add a new pipe every 2 seconds (at 30 FPS)

score = 0
game_over = False

# Clock
clock = pygame.time.Clock()

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if not game_over:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    velocity = -7
        else:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # Reset game
                game_over = False
                score = 0
                bird_y = screen_height // 2
                velocity = 0
                pipes = []

    if not game_over:
        # Update bird's vertical position
        velocity += gravity
        bird_y += velocity

        # Check for collisions with ground or ceiling
        if bird_y < 0:
            bird_y = 0
        elif bird_y + bird_height > screen_height:
            bird_y = screen_height - bird_height
            game_over = True

        # Update pipes
        if pipe_timer >= pipe_add_interval:
            # Add new pipe
            upper_pipe_y = random.randint(50, screen_height - 50 - pipe_gap)
            new_pipe = {
                'upper': pygame.Rect(screen_width, 0, pipe_width, upper_pipe_y),
                'lower': pygame.Rect(screen_width, upper_pipe_y + pipe_gap, pipe_width, screen_height - (upper_pipe_y + pipe_gap)),
                'scored': False
            }
            pipes.append(new_pipe)
            pipe_timer = 0
        else:
            pipe_timer += 1

        # Move pipes to the left
        for pipe in pipes:
            pipe['upper'].x += pipe_velocity
            pipe['lower'].x += pipe_velocity

        # Remove off-screen pipes
        pipes = [p for p in pipes if p['upper'].x + p['upper'].width > 0]

        # Check collisions with pipes
        bird_rect = pygame.Rect(bird_x, bird_y, bird_width, bird_height)
        for pipe in pipes:
            if bird_rect.colliderect(pipe['upper']) or bird_rect.colliderect(pipe['lower']):
                game_over = True

        # Update score
        for pipe in pipes:
            if bird_x > pipe['upper'].x + pipe_width and not pipe['scored']:
                score += 1
                pipe['scored'] = True

    # Drawing
    screen.fill(background_color)

    if not game_over:
        # Draw bird
        pygame.draw.rect(screen, bird_color, (bird_x, bird_y, bird_width, bird_height))

        # Draw pipes
        for pipe in pipes:
            pygame.draw.rect(screen, pipe_color, pipe['upper'])
            pygame.draw.rect(screen, pipe_color, pipe['lower'])

        # Draw score
        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(text, (10, 10))
    else:
        # Game over screen
        font = pygame.font.Font(None, 72)
        text = font.render("Game Over!", True, (255, 0, 0))
        screen.blit(text, (screen_width // 4, screen_height // 3))
        font = pygame.font.Font(None, 36)
        text = font.render(f"Final Score: {score}", True, (255, 255, 255))
        screen.blit(text, (screen_width // 3, screen_height // 2))
        text = font.render("Press Space to Restart", True, (255, 255, 255))
        screen.blit(text, (screen_width // 3, screen_height // 2 + 50))

    # Update display
    pygame.display.flip()
    clock.tick(30)

pygame.quit()