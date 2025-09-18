import random
from collections import deque, namedtuple
from typing import Any, Deque, List

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayMemory(object):

    def __init__(self, capacity: int):
        self.memory: Deque[Transition] = deque([], maxlen=capacity)

    def push(self, *args: Any) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
