import random
from collections import deque
from typing import Deque, List

import numpy as np

from challenger_bot.reinforcement_learning.replay.experience import Experience, InsufficientExperiencesError


class ExperienceReplayHandler:
    """
    TODO: Look at deque: https://github.com/keras-rl/keras-rl/issues/165
    """

    def __init__(self, size: int = 100000, batch_size: int = 128):
        self.size = size
        self.batch_size = batch_size
        self.memory: Deque[Experience] = deque(maxlen=size)

    def record_experience(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self):
        if self.batch_size > len(self.memory):
            raise InsufficientExperiencesError
        return random.sample(self.memory, self.batch_size)

    def generator(self) -> List[Experience]:
        batch = self.sample()

        for experience in batch:
            yield experience
