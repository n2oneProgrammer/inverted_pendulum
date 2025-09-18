from typing import Tuple

import pygame
from Box2D import b2Body, b2Color, b2Fixture

from environment.params import HEIGHT, PPM, WIDTH


def world_to_screen(pos: Tuple[int, int]) -> Tuple[int, int]:
    return int(pos[0] * PPM + WIDTH / 2), int(HEIGHT - pos[1] * PPM)


def screen_to_world(pos: Tuple[int, int]) -> Tuple[float, float]:
    return (pos[0] - WIDTH / 2) / PPM, (HEIGHT - pos[1]) / PPM


def draw_polygon(
    surface: pygame.Surface, body: b2Body, fixture: b2Fixture, color: b2Color
) -> None:
    shape = fixture.shape
    vertices = [body.transform * v for v in shape.vertices]
    vertices = [world_to_screen(v) for v in vertices]
    pygame.draw.polygon(surface, color, vertices)
