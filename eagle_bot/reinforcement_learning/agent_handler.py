import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Callable

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

        self.training_rewards = []
        self.episode = 0
        self.evaluation_rewards = []
        self.evaluations: int = 0

        self.evaluation: bool = False

        self.current_shot_total_reward = 0

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
        if done:
            self.end_episode_cleanup()

        if self.evaluation:
            action = self.current_agent.train_with_get_output(state, reward=reward, done=done, evaluation=True)
        else:
            action = self.current_agent.train_with_get_output(state, reward=reward, done=done)

        if done:
            return SimpleControllerState(), done

        controller_state = self.get_controller_state_from_actions(action)

        # print(controller_state.__dict__)
        self.draw()
        return controller_state, done

    def get_state(self, rb_tick: RigidBodyTick, packet: GameTickPacket) -> np.ndarray:
        ball_state = rb_tick.ball.state
        car_state = rb_tick.players[0].state

        dx = car_state.location.x - ball_state.location.x
        dy = car_state.location.y - ball_state.location.y
        dz = car_state.location.z - ball_state.location.z
        array = np.array([
            dx, dy, dz,

            car_state.velocity.x / 2000,
            car_state.velocity.y / 2000,
            car_state.velocity.z / 2000,
            car_state.rotation.x,
            car_state.rotation.y,
            car_state.rotation.z,
            car_state.rotation.w,
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
        else:
            car_ball_separation = ((car_location.x - ball_location.x) ** 2 +
                                   (car_location.y - ball_location.y) ** 2 +
                                   (car_location.z - ball_location.z) ** 2) ** 0.5
            if car_ball_separation > 400:
                done = True
        if done:
            return 0, done
        return 0.1, done

    def create_agent(self, trained_model_filepaths: Tuple[Path, Path] = None) -> DDPGAgent:
        INPUTS = 17
        OUTPUTS = 4  # pitch, yaw, roll, boost
        if trained_model_filepaths:
            actor_model = DenseModel(INPUTS, OUTPUTS, load_from_filepath=str(trained_model_filepaths[0]))
            critic_model = BaseCriticModel(INPUTS, OUTPUTS, learning_rate=1e-3,
                                           load_from_filepath=str(trained_model_filepaths[1]))
            print(f"Loaded trained actor: {trained_model_filepaths[0].name} and "
                  f"trained critic: {trained_model_filepaths[1].name}")
        else:
            actor_model = DenseModel(inputs=INPUTS, outputs=OUTPUTS, layer_nodes=(256, 256, 256),
                                     inner_activation='relu', output_activation='tanh')
            critic_model = BaseCriticModel(inputs=INPUTS, outputs=OUTPUTS, learning_rate=1e-3)
        agent = DDPGAgent(
            actor_model, critic_model=critic_model,
            exploration=OrnsteinUhlenbeckAndEpsilonGreedy(theta=0.15, sigma=0.05, dt=1 / 120, size=OUTPUTS,
                                                          epsilon_actions=1),
            actor_learning_rate=1e-3
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
            print(f"Evaluation {self.evaluations} reward: {self.current_shot_total_reward:.2f}")
        else:
            self.training_rewards.append(self.current_shot_total_reward)
            print(f"Shot {self.episode} reward: {self.current_shot_total_reward:.2f}")
        self.current_shot_total_reward = 0
        self.episode += 1

        if self.episode % 50 == 1:
            self.evaluation = True
            if self.episode % 200 == 1 and self.episode > 500:
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
        x_offset = 675
        renderer.draw_rect_2d(x_offset + 140, 95, 440, 50, True, renderer.create_color(80, 0, 0, 0))
        renderer.draw_string_2d(x_offset + 145, 100, x_scale, y_scale, f"EPISODE: {self.episode}",
                                renderer.white())

        renderer.draw_rect_2d(x_offset + 140, 145, 440, 50, True, renderer.create_color(80, 0, 0, 0))
        renderer.draw_string_2d(x_offset + 145, 150, x_scale, y_scale,
                                f"EP REWARD: {self.current_shot_total_reward:.1f}",
                                renderer.white())

        # renderer.draw_rect_2d(x_offset + 140, 145, 440, 50, True, renderer.create_color(80, 0, 0, 0))
        # renderer.draw_string_2d(x_offset + 145, 150, x_scale, y_scale,
        #                         f"GOALS: {self.goals} ({self.goals - self.ghost_goals} ML)",
        #                         renderer.white())
        #
        # renderer.draw_rect_2d(x_offset + 140, 195, 440, 50, True, renderer.create_color(80, 0, 0, 0))
        # renderer.draw_string_2d(x_offset + 145, 200, x_scale, y_scale,
        #                         f"EVAL: {self.evaluation_goals} / {self.evaluations}",
        #                         renderer.white())

        if self.evaluation:
            renderer.draw_string_2d(x_offset + 100, 20, 5, 5, "EVALUATING",
                                    renderer.white() if int(time.time() * 3) % 2 == 0 else renderer.grey())

        renderer.end_rendering()
