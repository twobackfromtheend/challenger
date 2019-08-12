from pathlib import Path
from typing import Union, Optional

import numpy as np


class BaseAgent:
    def train_with_get_output(
            self, state: np.ndarray, reward: float, done: bool,
            evaluation: bool = False
    ) -> Union[np.ndarray, None]:
        """
        Conducts a training step while getting an action (if not done).
        :param state:
        :param reward:
        :param done:
        :param evaluation:
        :return action if not done, else None:
        """
        raise NotImplementedError

    @staticmethod
    def initialise_from_checkpoint(folder: Path, *args):
        """
        Loads the agent from a saved checkpoint
        :param folder:
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, folder: Path):
        """
        Saves a checkpoint from which it can be reloaded using `load_checkpoint`
        :param folder:
        :return:
        """
        raise NotImplementedError
