import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Callable

import math
import numpy as np
from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.structures.rigid_body_struct import RigidBodyTick

from eagle_bot.reinforcement_learning.agent import DDPGAgent
from eagle_bot.reinforcement_learning.exploration import OrnsteinUhlenbeckAndEpsilonGreedy
from eagle_bot.reinforcement_learning.model.base_critic_model import BaseCriticModel
from eagle_bot.reinforcement_learning.model.dense_model import DenseModel
from eagle_bot.reinforcement_learning.model_handler import get_model_savepath, get_latest_model

if TYPE_CHECKING:
    from rlbot.utils.rendering.rendering_manager import RenderingManager


class AgentHandler:
    trained_models_folder = Path(__file__).parent / "trained_models"

    def __init__(self, renderer: 'RenderingManager'):
        self.renderer = renderer

        self.current_agent: Optional[DDPGAgent] = None

        self.training_rewards = deque([0], maxlen=200)
        self.training_durations = deque([0], maxlen=200)

        self.best_episode: Tuple[int, float, float] = (0, 0.0, 0.0)  # Episode, reward, duration
        self.episode = 0
        self.evaluation_rewards = []
        self.evaluations: int = 0

        self.evaluation: bool = False

        self.current_shot_total_reward = 0
        self.current_shot_duration = 0
        self.previous_boosts = deque([0, 0, 0], maxlen=3)

    def get_agent(self):
        # DESIRED_MODEL = "_20190411-113911.h5"
        DESIRED_MODEL = ""
        trained_actor_filepath = get_latest_model(self.trained_models_folder, "actor*" + DESIRED_MODEL)
        trained_critic_filepath = get_latest_model(self.trained_models_folder, "critic*" + DESIRED_MODEL)

        if trained_actor_filepath is not None and trained_critic_filepath is not None and \
                trained_actor_filepath.is_file() and trained_critic_filepath.is_file():
            self.current_agent = self.create_agent((trained_actor_filepath, trained_critic_filepath))
        else:
            self.current_agent = self.create_agent()

    def is_setup(self) -> bool:
        return self.current_agent is not None

    def challenger_tick(self, packet: GameTickPacket,
                        get_rb_tick: Callable[[], RigidBodyTick],
                        previous_packet: GameTickPacket) -> Tuple[SimpleControllerState, bool]:
        if self.current_agent is None:
            raise Exception("Not loaded agent.")

        # print("train")
        rb_tick = get_rb_tick()

        state = self.get_state(rb_tick, packet)
        reward, done = self.get_reward_and_done(rb_tick, packet)
        self.current_shot_total_reward += reward
        self.current_shot_duration += packet.game_info.game_time_remaining - previous_packet.game_info.game_time_remaining
        if done:
            self.end_episode_cleanup()

        if self.evaluation:
            action = self.current_agent.train_with_get_output(state, reward=reward, done=done, evaluation=True)
        else:
            action = self.current_agent.train_with_get_output(state, reward=reward, done=done)

        if done:
            return SimpleControllerState(), done

        controller_state = self.get_controller_state_from_actions(action)
        self.previous_boosts.append(int(controller_state.boost))

        # print(controller_state.__dict__)
        self.draw()
        return controller_state, done

    def get_state(self, rb_tick: RigidBodyTick, packet: GameTickPacket) -> np.ndarray:
        ball_state = rb_tick.ball.state
        car_state = rb_tick.players[0].state

        dx = car_state.location.x - ball_state.location.x
        dy = car_state.location.y - ball_state.location.y
        dz = car_state.location.z - ball_state.location.z

        w, x, y, z = car_state.rotation.w, car_state.rotation.x, car_state.rotation.y, car_state.rotation.z

        phi = math.atan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
        theta = math.asin(2 * (w * y - z * x))
        psi = math.atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))

        car_input = rb_tick.players[0].input

        array = np.array([
            dx, dy, dz,

            car_input.boost,
            car_input.pitch,
            car_input.yaw,
            car_input.roll,

            self.previous_boosts[0],
            self.previous_boosts[1],
            self.previous_boosts[2],


            car_state.location.x / 2000,
            car_state.location.y / 2000,
            car_state.location.z / 2000,

            car_state.velocity.x / 2000,
            car_state.velocity.y / 2000,
            car_state.velocity.z / 2000,
            # car_state.rotation.x,
            # car_state.rotation.y,
            # car_state.rotation.z,
            # car_state.rotation.w,

            math.sin(phi),
            math.cos(phi),
            math.sin(theta),
            math.cos(theta),
            math.sin(psi),
            math.cos(psi),

            car_state.angular_velocity.x,
            car_state.angular_velocity.y,
            car_state.angular_velocity.z,

            ball_state.location.z,
            ball_state.velocity.x / 2000,
            ball_state.velocity.y / 2000,
            ball_state.velocity.z / 2000,

        ])
        return array

    def get_reward_and_done(self, rb_tick: RigidBodyTick, packet: GameTickPacket) -> Tuple[float, bool]:
        done = False

        ball_location = rb_tick.ball.state.location
        car_location = rb_tick.players[0].state.location

        if ball_location.z < 500:
            done = True
        elif abs(ball_location.x) > 2000 or abs(ball_location.y) > 2000:
            done = True
        else:
            car_ball_separation = ((car_location.x - ball_location.x) ** 2 +
                                   (car_location.y - ball_location.y) ** 2 +
                                   (car_location.z - ball_location.z) ** 2) ** 0.5
            if car_ball_separation > 400:
                done = True
        if done:
            return -1, done
        return 0.1, done

    def create_agent(self, trained_model_filepaths: Tuple[Path, Path] = None) -> DDPGAgent:
        INPUTS = 29
        OUTPUTS = 4  # pitch, yaw, roll, boost
        if trained_model_filepaths:
            actor_model = DenseModel(INPUTS, OUTPUTS, load_from_filepath=str(trained_model_filepaths[0]))
            critic_model = BaseCriticModel(INPUTS, OUTPUTS, learning_rate=1e-5,
                                           load_from_filepath=str(trained_model_filepaths[1]))
            print(f"Loaded trained actor: {trained_model_filepaths[0].name} and "
                  f"trained critic: {trained_model_filepaths[1].name}")
        else:
            actor_model = DenseModel(inputs=INPUTS, outputs=OUTPUTS, layer_nodes=(256, 256, 256),
                                     inner_activation='relu', output_activation='tanh')
            critic_model = BaseCriticModel(inputs=INPUTS, outputs=OUTPUTS, learning_rate=1e-5)
        agent = DDPGAgent(
            actor_model, critic_model=critic_model,
            exploration=OrnsteinUhlenbeckAndEpsilonGreedy(theta=0.15, sigma=0.1, dt=1 / 60, size=OUTPUTS,
                                                          epsilon_actions=1),
            actor_learning_rate=1e-5
        )
        print("Created agent")
        return agent

    @staticmethod
    def get_controller_state_from_actions(action: np.ndarray) -> SimpleControllerState:
        controls = action.clip(-1, 1)
        controller_state = SimpleControllerState()
        controller_state.pitch = controls[0]
        controller_state.yaw = controls[1]
        controller_state.roll = controls[2]
        controller_state.boost = bool(controls[3] >= 0)
        return controller_state

    def end_episode_cleanup(self):
        if self.evaluation:
            self.evaluation_rewards.append(self.current_shot_total_reward)
            self.evaluations += 1
            print(f"Evaluation {self.evaluations} reward: {self.current_shot_total_reward:.1f} "
                  f"({self.current_shot_duration:.2f}s)")
        else:
            self.training_rewards.append(self.current_shot_total_reward)
            self.training_durations.append(self.current_shot_duration)
            print(f"Shot {self.episode} reward: {self.current_shot_total_reward:.1f} "
                  f"({self.current_shot_duration:.2f}s)")

        if self.current_shot_duration > self.best_episode[2]:
            self.best_episode = (self.episode, self.current_shot_total_reward, self.current_shot_duration)

        self.current_shot_total_reward = 0
        self.current_shot_duration = 0
        self.previous_boosts[0] = 0
        self.previous_boosts[1] = 0
        self.previous_boosts[2] = 0
        self.episode += 1

        if self.episode % 50 == 1:
            self.evaluation = True
            if self.episode % 1000 == 1 and self.episode > 500:
                print(f"Saving models: {self.trained_models_folder}")
                actor_savepath = get_model_savepath(self.trained_models_folder, f"actor_{self.episode}")
                critic_savepath = get_model_savepath(self.trained_models_folder, f"critic_{self.episode}")
                self.current_agent.actor_model.save_model(actor_savepath)
                self.current_agent.critic_model.save_model(critic_savepath)
        else:
            self.evaluation = False

    def draw(self):
        renderer = self.renderer
        renderer.begin_rendering('agent_handler')
        x_scale = y_scale = 3
        rect_width = 450
        x_offset = 15
        x_offset_ = x_offset + 5
        renderer.draw_rect_2d(x_offset, 95, rect_width, 50, True, renderer.create_color(80, 0, 0, 0))
        renderer.draw_string_2d(x_offset_, 100, x_scale, y_scale, f"EPISODE: {self.episode}",
                                renderer.white())

        renderer.draw_rect_2d(x_offset, 145, rect_width, 50, True, renderer.create_color(80, 0, 0, 0))
        renderer.draw_string_2d(x_offset_, 150, x_scale, y_scale,
                                f"EP REWARD: {self.current_shot_total_reward:.1f}",
                                renderer.white())

        renderer.draw_rect_2d(x_offset, 195, rect_width, 50, True, renderer.create_color(80, 0, 0, 0))
        renderer.draw_string_2d(x_offset_, 200, x_scale, y_scale,
                                f"EP TIME: {self.current_shot_duration:.2f}",
                                renderer.white())

        renderer.draw_rect_2d(x_offset, 245, rect_width, 50, True, renderer.create_color(80, 0, 0, 0))
        renderer.draw_string_2d(x_offset_, 250, x_scale, y_scale,
                                f"AVG: R: {sum(self.training_rewards) / len(self.training_rewards):.1f}, "
                                f"D: {sum(self.training_durations) / len(self.training_durations):.1f}",
                                renderer.white())

        renderer.draw_rect_2d(x_offset, 295, rect_width, 50, True, renderer.create_color(80, 0, 0, 0))
        renderer.draw_string_2d(x_offset_, 300, x_scale, y_scale,
                                f"BEST: EP: {self.best_episode[0]},",
                                renderer.white())
        renderer.draw_rect_2d(x_offset, 345, rect_width, 50, True, renderer.create_color(80, 0, 0, 0))
        renderer.draw_string_2d(x_offset_, 350, x_scale, y_scale,
                                f"     R: {self.best_episode[1]:.1f}, D: {self.best_episode[2]:.1f}",
                                renderer.white())

        if self.evaluation:
            renderer.draw_string_2d(x_offset, 20, 5, 5, "EVALUATING",
                                    renderer.white() if int(time.time() * 3) % 2 == 0 else renderer.grey())

        renderer.end_rendering()
