from functools import reduce

from DQN_model import DQN
from matplotlib import pyplot as plt

from environment.environment import InvertedPendulumEnv


def calc_height(env: InvertedPendulumEnv) -> float:
    return env.pole[-1].body.position.y - env.cart.body.position.y  # type: ignore[no-any-return]


env = InvertedPendulumEnv()

input_dim = env.observation_space.shape[0]
output_dim = 3
dqn = DQN(input_dim, output_dim, load_model="models/3_pos_out.pth")

state, info = env.reset()
steps = 0
heights: list[float] = []
while steps < 1e3:
    action = dqn.get_action(state, training=False) - 1
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state = next_state
    steps += 1
    heights.append(calc_height(env))
    if steps % 1000 == 0:
        print("steps:", steps)
env.close()

plt.plot(heights)
plt.xlabel("steps")
plt.ylabel("height")
plt.savefig("docs/plot.png")
plt.show()
print(reduce(lambda x, y: x + (1 + 0.2 - y) ** 2 * 1, heights, 0.0) / len(heights))
