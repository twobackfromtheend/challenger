import gym

from eagle_bot.reinforcement_learning.agent import DDPGAgent
from eagle_bot.reinforcement_learning.exploration import OrnsteinUhlenbeckAndEpsilonGreedy
from eagle_bot.reinforcement_learning.model.base_critic_model import BaseCriticModel

gym.envs.register(
    id='PendulumTimeSensitive-v0',
    entry_point='gym.envs.classic_control:PendulumEnv',
    max_episode_steps=200
)
env = gym.make('PendulumTimeSensitive-v0')

from eagle_bot.reinforcement_learning.model.dense_model import DenseModel

actor_model = DenseModel(inputs=3, outputs=1, layer_nodes=(48, 48), learning_rate=3e-3,
                         inner_activation='relu', output_activation='tanh')
critic_model = BaseCriticModel(inputs=3, outputs=1)
agent = DDPGAgent(
    actor_model, critic_model=critic_model,
    exploration=OrnsteinUhlenbeckAndEpsilonGreedy(theta=0.15, sigma=0.3),
    actor_learning_rate=3e-3
)

# Emulate get_output
episode_count = 0
while True:
    state = env.reset()
    total_reward = 0
    reward = 0
    done = False
    while True:
        evaluation = episode_count % 50 == 1
        if done:
            agent.train_with_get_output(state, reward=reward, done=True, evaluation=evaluation)
            break
        action = agent.train_with_get_output(state, reward=reward, done=False, evaluation=evaluation)
        state, reward, done, info = env.step(action)
        total_reward += reward
        env.render()

    print(f"Episode: {episode_count}: reward: {total_reward}")
    episode_count += 1
