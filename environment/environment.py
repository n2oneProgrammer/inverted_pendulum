import math
import time
from typing import Any, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
from Box2D import b2Vec2, b2World
from gymnasium import spaces

from environment.cart import CART_HEIGHT, Cart  # type: ignore[attr-defined]
from environment.params import (
    BG_COLOR,
    GROUND_COLOR,
    HEIGHT,
    POLE_LENGTH,
    POS_ITERS,
    PPM,
    TEXT_COLOR,
    TIMESTEP,
    VEL_ITERS,
    WIDTH,
)
from environment.pole import Pole


class InvertedPendulumEnv(gym.Env):  # type: ignore[misc]
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        title: str = "Inverted Pendulum",
        gravity: Tuple[float, float] = (0, -9.81),
        n_pole: int = 1,
        render_mode: Optional[str] = None,
    ):
        super(InvertedPendulumEnv, self).__init__()
        self.cart: Cart = None  # type: ignore[assignment]
        self.pole: List[Pole] = []
        self.gravity: b2Vec2 = b2Vec2(*gravity)
        self.n_pole: int = n_pole
        if render_mode in ["human", "rgb_array"]:
            pygame.init()
            self.screen: pygame.Surface = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption(title)
            self.clock: pygame.time.Clock = pygame.time.Clock()
            self.font: pygame.font.Font = pygame.font.SysFont("consolas", 18)
        self.world: b2World = b2World(gravity=self.gravity, doSleep=True)
        self.action_space: spaces.Box = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )

        self.observation_space: spaces.Box = spaces.Box(
            low=0, high=1, shape=(2 + 2 * n_pole,), dtype=np.float32
        )

        self.state: np.ndarray = np.array([], dtype=np.float32)
        self.render_mode: Optional[str] = render_mode
        self.start_time: float = time.time()
        self.reset()

    def create_state(self) -> np.ndarray:
        result = [self.cart.body.position.x, self.cart.body.linearVelocity.x]

        for pole in self.pole:
            result.extend(
                [
                    (pole.body.angle * 180 / math.pi + 180) % 360 - 180,
                    pole.body.angularVelocity,
                ]
            )
        return np.array(result)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        for body in list(self.world.bodies):
            self.world.DestroyBody(body)
        anchor = self.world.CreateStaticBody(position=(0, 2))
        self.cart = Cart(self.world, anchor)
        prev = self.cart.body
        prev_height = CART_HEIGHT
        self.pole = []
        for i in range(self.n_pole):
            pole = Pole(self.world, prev, prev_height)
            prev = pole.body
            prev_height = POLE_LENGTH
            self.pole.append(pole)

        return self.create_state(), {}

    def count_reward(self) -> float:
        x = self.cart.body.position.x
        reward: float = 0
        for pole in self.pole:
            angle = (pole.body.angle * 180 / math.pi + 180) % 360 - 180
            reward += 1 - abs(angle) / 45
            if abs(angle) > 10:
                break

        reward -= abs(x) / 2
        return reward

    def step(
        self, action: float
    ) -> tuple[np.ndarray, float, bool, bool, dict[Any, Any]]:
        self.cart.apply_force(action)
        self.world.Step(TIMESTEP, VEL_ITERS, POS_ITERS)
        self.world.ClearForces()
        reward = self.count_reward()

        terminated = False
        truncated = False

        return self.create_state(), reward, terminated, truncated, {}

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.screen.fill(BG_COLOR)
            ground_y = 0.2
            pygame.draw.rect(
                self.screen,
                GROUND_COLOR,
                (0, HEIGHT - int(ground_y * PPM), WIDTH, int(ground_y * PPM)),
            )
            self.cart.draw(self.screen)
            for pole in self.pole:
                pole.draw(self.screen)

            fps = int(self.clock.get_fps())
            r = self.font.render(f"FPS = {fps}", True, TEXT_COLOR)
            self.screen.blit(r, (10, HEIGHT - 30))
            pygame.display.flip()
            self.clock.tick(200)
            if self.render_mode == "rgb_array":
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(self.screen)), (1, 0, 2)
                )
        return None

    def close(self) -> None:
        pygame.quit()
