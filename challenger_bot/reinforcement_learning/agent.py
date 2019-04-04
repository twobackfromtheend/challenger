import numpy as np
from tensorflow.python.keras.optimizers import Adam

from challenger_bot.reinforcement_learning.model.base_critic_model import BaseCriticModel
from challenger_bot.reinforcement_learning.model.base_model import BaseNNModel
from challenger_bot.reinforcement_learning.replay.experience import InsufficientExperiencesError
from challenger_bot.reinforcement_learning.replay.replay_handler import ExperienceReplayHandler


class DDPGAgent:
    def __init__(self,
                 actor_model: BaseNNModel,
                 critic_model: BaseCriticModel,
                 exploration,
                 ):
        self.actor_model = actor_model
        self.critic_model = critic_model

        self.target_actor_model = actor_model.create_copy()
        self.target_critic_model = critic_model.create_copy()

        self.exploration = exploration
        self.replay_handler = ExperienceReplayHandler()

        self.last_state: np.ndarray = np.zeros(actor_model.inputs)
        self.last_action: np.ndarray = np.zeros(actor_model.outputs)

        self.discount_rate = 0.95

        self.actor_train_fn = self.critic_model.get_actor_train_fn(self.actor_model, Adam(1e-3))

    def train_with_get_output(self, state: np.ndarray, reward: float, done: bool):
        action = self.get_action(state)

        self.replay_handler.record_experience(self.last_state, self.last_action, reward, state, done)

        self.update_target_models(True, 0.01)

        try:
            self.experience_replay()
        except InsufficientExperiencesError:
            pass

        self.last_state = state
        self.last_action = action
        return action

    def get_action(self, state: np.ndarray):
        action = self.actor_model.predict(state.reshape((1, -1))).flatten()
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


if __name__ == '__main__':
    import gym
    from challenger_bot.reinforcement_learning.exploration.ornstein_uhlenbeck import OrnsteinUhlenbeck

    gym.envs.register(
        id='PendulumTimeSensitive-v0',
        entry_point='gym.envs.classic_control:PendulumEnv',
        max_episode_steps=200
    )
    env = gym.make('PendulumTimeSensitive-v0')

    from challenger_bot.reinforcement_learning.model.dense_model import DenseModel

    actor_model = DenseModel(inputs=3, outputs=1, layer_nodes=(48, 48), learning_rate=3e-3,
                             inner_activation='relu', output_activation='tanh')
    critic_model = BaseCriticModel(inputs=3, outputs=1)
    agent = DDPGAgent(
        actor_model, critic_model=critic_model,
        exploration=OrnsteinUhlenbeck(theta=0.15, sigma=0.3),

    )

    # Emulate get_output
    episode_count = 0
    while True:
        state = env.reset()
        total_reward = 0
        reward = 0
        done = False
        while True:
            if done:
                agent.train_with_get_output(state, reward=reward, done=False)
                break
            action = agent.train_with_get_output(state, reward=reward, done=False)
            state, reward, done, info = env.step(action)
            total_reward += reward
            env.render()

        print(f"Episode: {episode_count}: reward: {total_reward}")
        episode_count += 1
