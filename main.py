import pygame
import torch
from gymnasium.wrappers import TimeLimit
from matplotlib import pyplot as plt
from pygame import K_ESCAPE, KEYDOWN, QUIT

from DQN_model import DQN
from environment.environment import InvertedPendulumEnv

training_period = 250

env = InvertedPendulumEnv(render_mode="human")
# env = gymnasium.make("CartPole-v1", render_mode="human")
# env = RecordVideo(
#     env,
#     video_folder="videos",
#     name_prefix="training",
#     episode_trigger=lambda x: True
# )

# Zbieranie statystyk
# env = RecordEpisodeStatistics(env)
env = TimeLimit(env, max_episode_steps=1000)
input_dim = env.observation_space.shape[0]
output_dim = 3
print(input_dim, output_dim)
dqn = DQN(input_dim, output_dim)

state, info = env.reset()
print(state)
max_episodes: float = 1e6
episode: float = 0
reward_sum: float = 0
rewards: list[float] = []
while episode < max_episodes:
    episode += 1
    for event in pygame.event.get():
        if event.type == QUIT:
            break
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                episode = max_episodes

    action = dqn.get_action(state, training=True)
    # action -= 1
    next_state, reward, terminated, truncated, info = env.step(action - 1)
    reward_sum += reward
    done = terminated or truncated
    dqn.update_model(state, action, reward, next_state, done)
    state = next_state
    env.render()
    if done:
        observation, _ = env.reset()
        dqn.update_epsilon()
        print(episode, info, dqn.epsilon, reward_sum)
        rewards.append(reward_sum)
        reward_sum = 0
        torch.save(dqn.target_net, "last_model.pth")

# dqn.target_net.save_weights("dqn_model.h5")
env.close()
torch.save(dqn.target_net, "full_model.pth")
plt.plot(rewards)
plt.show()
