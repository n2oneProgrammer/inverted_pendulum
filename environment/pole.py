import math

import pygame
from Box2D import (
    b2Body,
    b2FixtureDef,
    b2PolygonShape,
    b2RevoluteJointDef,
    b2Vec2,
    b2World,
)

from environment.helpers import draw_polygon
from environment.params import (
    NUDGE_IMPULSE,
    POLE_COLOR,
    POLE_LENGTH,
    POLE_MASS,
    POLE_WIDTH,
)


class Pole:
    def __init__(self, world: b2World, cart_body: b2Body, cart_height: float):
        fixture = b2FixtureDef(
            shape=b2PolygonShape(box=(POLE_WIDTH / 2.0, POLE_LENGTH / 2.0)),
            density=POLE_MASS / (POLE_WIDTH * POLE_LENGTH),
            friction=0.1,
        )
        fixture.filter.groupIndex = -1
        self.body = world.CreateDynamicBody(
            position=(
                cart_body.position.x,
                cart_body.position.y + (cart_height / 2.0) + POLE_LENGTH / 2.0,
            ),
            fixtures=fixture,
        )
        world.CreateJoint(
            b2RevoluteJointDef(
                bodyA=cart_body,
                bodyB=self.body,
                localAnchorA=(0.0, cart_height / 2.0),
                localAnchorB=(0.0, -POLE_LENGTH / 2.0),
                enableLimit=False,
                enableMotor=False,
            )
        )
        self.body.angle = 180 * math.pi / 180

    def nudge(self) -> None:
        world_center = self.body.worldCenter
        nudge_point = (world_center.x, world_center.y + POLE_LENGTH / 2.0)
        self.body.ApplyLinearImpulse(
            impulse=b2Vec2(0, NUDGE_IMPULSE), point=nudge_point, wake=True
        )

    def draw(self, screen: pygame.Surface) -> None:
        for fixture in self.body.fixtures:
            draw_polygon(screen, self.body, fixture, POLE_COLOR)
