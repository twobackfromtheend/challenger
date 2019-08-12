import random
from typing import Union, Optional

import numpy as np

from eagle_bot.reinforcement_learning.exploration import OrnsteinUhlenbeckAndEpsilonGreedy
from eagle_bot.reinforcement_learning.model.base_model import BaseModel
from eagle_bot.reinforcement_learning.model.base_critic_model import BaseCriticModel
from eagle_bot.reinforcement_learning.replay.experience import InsufficientExperiencesError
from eagle_bot.reinforcement_learning.replay.replay_handler import ExperienceReplayHandler


class DDPGAgent:
    def __init__(self,
                 actor_model: BaseModel,
                 critic_model: BaseCriticModel,
                 exploration: OrnsteinUhlenbeckAndEpsilonGreedy,
                 actor_learning_rate: float
                 ):
        self.actor_model = actor_model
        self.critic_model = critic_model

        self.target_actor_model = actor_model.create_copy()
        self.target_critic_model = critic_model.create_copy()

        self.exploration = exploration
        self.replay_handler = ExperienceReplayHandler(size=1500000, batch_size=512, warmup=100000)

        self.last_state: np.ndarray = np.zeros(actor_model.inputs)
        self.last_action: np.ndarray = np.zeros(actor_model.outputs)

        self.discount_rate = 0.999  # https://www.wolframalpha.com/input/?i=ln(2)+%2F+(1+-+0.999)+%2F+60
        from tensorflow.python.keras.optimizers import Adam
        self.actor_train_fn = self.critic_model.get_actor_train_fn(self.actor_model, Adam(actor_learning_rate))

        self.i = 0

    def reset_last_state_and_action(self):
        self.last_state = np.zeros(self.actor_model.inputs)
        self.last_action = np.zeros(self.actor_model.outputs)

    def train_with_get_output(self, state: np.ndarray, reward: float, done: bool,
                              enforced_action: Optional[np.ndarray] = None,
                              evaluation: bool = False) -> Union[np.ndarray, None]:
        if random.random() < 0.3 or done:
            self.replay_handler.record_experience(self.last_state, self.last_action, reward, state, done)

        self.i += 1
        # if done:
        if self.i == 50 or done:
            self.update_target_models(True, 0.005)

            try:
                critic_loss = self.experience_replay()
                # from quicktracer import trace
                # trace(float(critic_loss))
            except InsufficientExperiencesError:
                pass
            self.i = 0

        if not done:
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
            # self.discount_rate = min(self.discount_rate + 0.00000001, 0.9993)

            self.reset_last_state_and_action()
            self.exploration.reset_states()

    def get_action(self, state: np.ndarray, evaluation: bool):
        action = self.actor_model.predict(state.reshape((1, -1))).flatten()
        if evaluation:
            return action

        return self.exploration.get_action(action)

    def experience_replay(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for experience in self.replay_handler.generator():
            states.append(experience.state)
            actions.append(experience.action)
            rewards.append(experience.reward)
            next_states.append(experience.next_state)
            dones.append(experience.done)

        # Converting to np.arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        targets = self.get_critic_targets(rewards, next_states, dones)

        states_with_actions = self.critic_model.create_input(states, actions)
        critic_loss = self.critic_model.train_on_batch(states_with_actions, targets)

        # Train actor
        actor_inputs = [states, True]  # True tells model that it's in training mode.
        action_values = self.actor_train_fn(actor_inputs)[0]  # actions not needed for anything.

        return critic_loss

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

        next_states_with_next_actions = self.target_critic_model.create_input(_next_states, target_next_actions)

        target_q_values = self.target_critic_model.predict_on_batch(next_states_with_next_actions).flatten()
        targets[done_false_indices] += gamma * target_q_values
        return targets

    def update_target_models(self, soft: bool, tau: float):
        """
        Update target model's weights with policy model's.
        If soft,
            theta_target <- tau * theta_policy + (1 - tau) * theta_target
            tau << 1.
        :param soft:
        :param tau: tau << 1 (recommended: 0.001)
        :return:
        """
        if soft:
            self.target_actor_model.model.set_weights(
                tau * np.array(self.actor_model.model.get_weights())
                + (1 - tau) * np.array(self.target_actor_model.model.get_weights())
            )
            self.target_critic_model.model.set_weights(
                tau * np.array(self.critic_model.model.get_weights())
                + (1 - tau) * np.array(self.target_critic_model.model.get_weights())
            )
        else:
            self.target_actor_model.model.set_weights(self.actor_model.model.get_weights())
            self.target_critic_model.model.set_weights(self.critic_model.model.get_weights())
