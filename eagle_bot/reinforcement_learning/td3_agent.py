import random
from pathlib import Path
from typing import Union, Optional, Tuple

import numpy as np

from eagle_bot.reinforcement_learning.agents.base_agent import BaseAgent
from eagle_bot.reinforcement_learning.exploration import OrnsteinUhlenbeckAndEpsilonGreedy
from eagle_bot.reinforcement_learning.model.base_critic_model import BaseCriticModel
from eagle_bot.reinforcement_learning.model.base_model import BaseModel
from eagle_bot.reinforcement_learning.model.dense_actor_model import DenseActorModel
from eagle_bot.reinforcement_learning.model.dense_critic_model import DenseCriticModel
from eagle_bot.reinforcement_learning.replay.baselines_replay_buffer import ReplayBuffer
from eagle_bot.reinforcement_learning.replay.experience import InsufficientExperiencesError


class TD3Agent(BaseAgent):
    def __init__(self,
                 actor_model: BaseModel,
                 critic_model: BaseCriticModel,
                 exploration: OrnsteinUhlenbeckAndEpsilonGreedy,
                 actor_learning_rate: float,
                 replay_handler: ReplayBuffer,
                 critic_model_2: BaseCriticModel = None,
                 target_actor_model: BaseModel = None,
                 target_critic_model: BaseCriticModel = None,
                 target_critic_model_2: BaseCriticModel = None,
                 target_policy_smoothing_noise: float = 0.1,
                 target_policy_smoothing_noise_clip: float = 0.2,
                 target_actions_clipping_range: Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]] = (-1, 1)
                 ):
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.critic_model_2 = critic_model.create_copy() if critic_model_2 is None else critic_model_2

        self.target_actor_model = actor_model.create_copy() if target_actor_model is None else target_actor_model
        self.target_critic_model = critic_model.create_copy() if target_critic_model is None else target_critic_model
        self.target_critic_model_2 = self.critic_model_2.create_copy() if target_critic_model_2 is None else target_critic_model_2

        self.exploration = exploration
        self.replay_handler = replay_handler

        self.last_state: np.ndarray = np.zeros(actor_model.inputs)
        self.last_action: np.ndarray = np.zeros(actor_model.outputs)

        self.discount_rate = 0.97
        from tensorflow.python.keras.optimizers import Adam
        self.actor_train_fn = self.critic_model.get_actor_train_fn(self.actor_model, Adam(actor_learning_rate))

        self.target_policy_smoothing_noise = target_policy_smoothing_noise
        self.target_policy_smoothing_noise_clip = target_policy_smoothing_noise_clip
        self.target_actions_clipping_range = target_actions_clipping_range

        self.i = 0
        self.j = 0  # Used for actor and target network training delay

    def reset_last_state_and_action(self):
        self.last_state = np.zeros(self.actor_model.inputs)
        self.last_action = np.zeros(self.actor_model.outputs)

    def train_with_get_output(self, state: np.ndarray, reward: float, done: bool,
                              enforced_action: Optional[np.ndarray] = None,
                              evaluation: bool = False) -> Union[np.ndarray, None]:
        if random.random() < 0.4:
            self.replay_handler.add(self.last_state, self.last_action, reward, state, done)

        self.i += 1
        if self.i == 3:
            self.i = 0

            self.j += 1
            if self.j == 2:
                self.j = 0
                self.update_target_models(0.01)
                try:
                    critic_loss = self.experience_replay(train_actor=True)
                except InsufficientExperiencesError:
                    pass
            else:
                try:
                    critic_loss = self.experience_replay(train_actor=False)
                except InsufficientExperiencesError:
                    pass

        if not done:
            self.discount_rate = min(self.discount_rate + 0.00001, 0.997)
            if evaluation:
                action = self.get_action(state, True)
            elif enforced_action is None:
                action = self.get_action(state, False)
            else:
                # action = self.exploration.get_action(enforced_action)
                action = enforced_action
            self.last_state = state
            self.last_action = action
            return action
        else:
            self.reset_last_state_and_action()
            self.exploration.reset_states()

    def get_action(self, state: np.ndarray, evaluation: bool):
        action = self.actor_model.predict(state.reshape((1, -1))).flatten()
        if evaluation:
            return action

        return self.exploration.get_action(action)

    def experience_replay(self, train_actor: bool):
        states, actions, rewards, next_states, dones = self.replay_handler.sample()

        targets = self.get_critic_targets(rewards, next_states, dones)

        states_with_actions = self.critic_model.create_input(states, actions)
        critic_loss_1 = self.critic_model.train_on_batch(states_with_actions, targets)
        critic_loss_2 = self.critic_model_2.train_on_batch(states_with_actions, targets)

        if train_actor:
            actor_inputs = [states, True]  # True tells model that it's in training mode.
            action_values = self.actor_train_fn(actor_inputs)[0]  # actions not needed for anything.

        return critic_loss_1 + critic_loss_2

    def get_critic_targets(self, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """
        Calculates targets based on done.
        If done,
            target = reward
        If not done,
            target = r + gamma * max(q_values(next_state))
        :param rewards:
        :param next_states:
        :param dones:
        :return: Targets - 1D np.array
        """
        # Targets initialised w/ done == True steps
        targets = rewards.copy()

        # Targets for done == False steps calculated with target network
        done_false_indices = dones == 0
        gamma = self.discount_rate

        # Below calculations only concern those where done is false
        _next_states = next_states[done_false_indices]
        target_next_actions = self.target_actor_model.predict_on_batch(_next_states)

        # Add target actions noise
        noise = np.clip(
            np.random.normal(scale=self.target_policy_smoothing_noise, size=target_next_actions.shape),
            -self.target_policy_smoothing_noise_clip, self.target_policy_smoothing_noise_clip
        )
        target_next_actions = np.clip(target_next_actions + noise,
                                      self.target_actions_clipping_range[0], self.target_actions_clipping_range[1])

        next_states_with_next_actions = self.target_critic_model.create_input(_next_states, target_next_actions)

        target_q_values_1 = self.target_critic_model.predict_on_batch(next_states_with_next_actions).flatten()
        target_q_values_2 = self.target_critic_model_2.predict_on_batch(next_states_with_next_actions).flatten()
        target_q_values = np.minimum(target_q_values_1, target_q_values_2)
        targets[done_false_indices] += gamma * target_q_values
        return targets

    def update_target_models(self, tau: float):
        """
        Update target model's weights with policy model's.
        If soft,
            theta_target <- tau * theta_policy + (1 - tau) * theta_target
            tau << 1.
        :param tau: tau << 1 (recommended: 0.001)
        :return:
        """
        self.target_actor_model.model.set_weights(
            tau * np.array(self.actor_model.model.get_weights())
            + (1 - tau) * np.array(self.target_actor_model.model.get_weights())
        )
        self.target_critic_model.model.set_weights(
            tau * np.array(self.critic_model.model.get_weights())
            + (1 - tau) * np.array(self.target_critic_model.model.get_weights())
        )
        self.target_critic_model_2.model.set_weights(
            tau * np.array(self.critic_model_2.model.get_weights())
            + (1 - tau) * np.array(self.target_critic_model_2.model.get_weights())
        )

    @staticmethod
    def initialise_from_checkpoint(folder: Path, inputs: int, outputs: int,
                                   critic_model_learning_rate: float, *args, **kwargs) -> 'TD3Agent':
        actor_model = DenseActorModel(inputs, outputs, load_from_filepath=str(folder / "actor_model.h5"))
        critic_model = DenseCriticModel(inputs, outputs, critic_model_learning_rate,
                                        load_from_filepath=str(folder / "critic_model.h5"))
        critic_model_2 = DenseCriticModel(inputs, outputs, critic_model_learning_rate,
                                          load_from_filepath=str(folder / "critic_model_2.h5"))

        target_actor_model = DenseActorModel(inputs, outputs, load_from_filepath=str(folder / "target_actor_model.h5"))
        target_critic_model = DenseCriticModel(inputs, outputs, critic_model_learning_rate,
                                               load_from_filepath=str(folder / "target_critic_model.h5"))
        target_critic_model_2 = DenseCriticModel(inputs, outputs, critic_model_learning_rate,
                                                 load_from_filepath=str(folder / "target_critic_model_2.h5"))
        return TD3Agent(
            *args, **kwargs,
            actor_model=actor_model, critic_model=critic_model, critic_model_2=critic_model_2,
            target_actor_model=target_actor_model, target_critic_model=target_critic_model,
            target_critic_model_2=target_critic_model_2
        )

    def save_checkpoint(self, folder: Path):
        self.actor_model.save_model("actor_model.h5")
        self.critic_model.save_model("critic_model.h5")
        self.critic_model_2.save_model("critic_model_2.h5")

        self.target_actor_model.save_model("target_actor_model.h5")
        self.target_critic_model.save_model("target_critic_model.h5")
        self.target_critic_model_2.save_model("target_critic_model_2.h5")
