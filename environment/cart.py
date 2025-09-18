from __future__ import annotations

from typing import Any

from Box2D import (
    b2Body,
    b2FixtureDef,
    b2PolygonShape,
    b2PrismaticJointDef,
    b2World,
)

from environment.helpers import draw_polygon, screen_to_world
from environment.params import (
    CART_COLOR,
    CART_HEIGHT,
    CART_MASS,
    CART_WIDTH,
    HORIZONTAL_FORCE,
    WIDTH,
)


class Cart:
    def __init__(self, world: b2World, anchor: b2Body) -> None:
        fixture: b2FixtureDef = b2FixtureDef(
            shape=b2PolygonShape(box=(CART_WIDTH / 2.0, CART_HEIGHT / 2.0)),
            density=CART_MASS / (CART_WIDTH * CART_HEIGHT),
            friction=0.6,
        )
        fixture.filter.groupIndex = -1

        self.body: b2Body = world.CreateDynamicBody(
            position=(0.0, 0.6 + CART_HEIGHT / 2.0),
            fixtures=fixture,
        )
        self.body.fixedRotation = True

        world.CreateJoint(
            b2PrismaticJointDef(
                bodyA=anchor,
                bodyB=self.body,
                localAnchorA=(0, 0),
                localAnchorB=(0, 0),
                localAxisA=(1, 0),
                enableLimit=True,
                lowerTranslation=screen_to_world((0, 0))[0] + CART_WIDTH / 2,
                upperTranslation=screen_to_world((WIDTH, 0))[0] - CART_WIDTH / 2,
                enableMotor=True,
                maxMotorForce=1000.0,
                motorSpeed=0.0,
            )
        )

    def apply_force(self, direction: float) -> None:
        force: float = HORIZONTAL_FORCE * 4 * direction
        self.body.ApplyForce(force=(force, 0), point=self.body.worldCenter, wake=True)

    def draw(self, screen: Any) -> None:
        for fixture in self.body.fixtures:
            draw_polygon(screen, self.body, fixture, CART_COLOR)
