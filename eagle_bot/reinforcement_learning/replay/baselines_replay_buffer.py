import numpy as np
import random

from eagle_bot.reinforcement_learning.replay.experience import InsufficientExperiencesError


class ReplayBuffer(object):
    def __init__(self, size: int, warmup: int, batch_size: int):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

        self._batch_size = batch_size
        self._warmup = warmup
        self._reached_max_size = False
        self._warmed_up = False

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

        if not self._reached_max_size and len(self._storage) == self._maxsize:
            self._reached_max_size = True
            print("Reached max replay memory size.")
        if not self._warmed_up and len(self._storage) >= self._warmup:
            self._warmed_up = True
            print("Replay memory warmed up")

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self):
        """Sample a batch of experiences.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        len_storage = len(self._storage)
        if self._batch_size > len_storage or self._warmup > len_storage:
            raise InsufficientExperiencesError
        idxes = [random.randint(0, len_storage - 1) for _ in range(self._batch_size)]
        return self._encode_sample(idxes)
