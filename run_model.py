import pygame
from pygame import K_ESCAPE, K_LEFT, K_RIGHT, KEYDOWN, QUIT

from DQN_model import DQN
from environment.environment import InvertedPendulumEnv

env = InvertedPendulumEnv(render_mode="rgb_array")

input_dim = env.observation_space.shape[0]
output_dim = 3
dqn = DQN(input_dim, output_dim, load_model="last_model.pth")

state, info = env.reset()
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
            break
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            if event.key == "R":
                state, _ = env.reset()
    keys = pygame.key.get_pressed()
    action_user = 0
    if keys[K_LEFT]:
        action_user = -1
    if keys[K_RIGHT]:
        action_user = 1
    action = dqn.get_action(state, training=False) - 1
    print(action)
    next_state, reward, terminated, truncated, info = env.step(
        action if action_user == 0 else action_user
    )
    done = terminated or truncated
    state = next_state
    env.render()
    if done:
        state, _ = env.reset()

env.close()
