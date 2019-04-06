import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Callable

import numpy as np
from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.structures.rigid_body_struct import RigidBodyTick

from challenger_bot.base_agent_handler import BaseAgentHandler
from challenger_bot.game_controller import GameState
from challenger_bot.reinforcement_learning.agent import DDPGAgent
from challenger_bot.reinforcement_learning.exploration.ornstein_uhlenbeck import OrnsteinUhlenbeck
from challenger_bot.reinforcement_learning.model.base_critic_model import BaseCriticModel
from challenger_bot.reinforcement_learning.model.dense_model import DenseModel

if TYPE_CHECKING:
    from challenger_bot.ghosts.ghost import GhostHandler
    from rlbot.utils.rendering.rendering_manager import RenderingManager

WAIT_TIME_BEFORE_HIT = 1
trained_models_folder = Path(__file__).parent / "trained_models"


class AgentHandler(BaseAgentHandler):
    def __init__(self, ghost_handler: 'GhostHandler', renderer: 'RenderingManager'):
        super().__init__(ghost_handler, renderer)

        self.trained_actor_filepath: Optional[Path] = None
        self.trained_critic_filepath: Optional[Path] = None

        self.current_agent: Optional[DDPGAgent] = None

        self.total_rewards = []
        self.episode = 0

        self.ghost_override: bool = False

        self.previous_game_state: Optional[GameState] = None
        self.current_shot_spawn_time: Optional[float] = None
        self.current_shot_total_reward = 0
        self.current_shot_start_time: Optional[float] = None

    def get_agent(self, round_: int):
        trained_actor_filepath = trained_models_folder / f"{round_}_actor.h5"
        trained_critic_filepath = trained_models_folder / f"{round_}_critic.h5"
        if trained_actor_filepath.is_file() and trained_critic_filepath.is_file():
            self.current_agent = self.create_agent((trained_actor_filepath, trained_critic_filepath))
        else:
            self.current_agent = self.create_agent()

    def is_setup(self) -> bool:
        return self.current_agent is not None

    def challenger_tick(self, packet: GameTickPacket, game_state: GameState,
                        get_rb_tick: Callable[[], RigidBodyTick]) -> SimpleControllerState:
        if self.current_agent is None:
            raise Exception("Not loaded agent.")

        if self.current_shot_spawn_time is None or \
                self.previous_game_state != GameState.ROUND_WAITING and game_state == GameState.ROUND_WAITING:
            self.current_shot_spawn_time = time.time()
            # print("spawn")
        current_time = time.time()
        # Only train if ongoing
        train_on_frame = False
        if game_state == GameState.ROUND_ONGOING:
            train_on_frame = True
            done = False
        elif self.previous_game_state == GameState.ROUND_ONGOING and game_state == GameState.ROUND_FINISHED:
            train_on_frame = True
            done = True
            self.current_shot_start_time = None
            # print("bye")

        if train_on_frame:
            # print("train")
            rb_tick = get_rb_tick()

            if self.current_shot_start_time is None:
                self.current_shot_start_time = current_time
            shot_time_elapsed = current_time - self.current_shot_start_time

            state = self.get_state(rb_tick, packet, shot_time_elapsed)
            reward = self.get_reward(rb_tick, packet, done)
            self.current_shot_total_reward += reward
            if done:
                self.end_episode_cleanup()

            if self.ghost_override:
                enforced_action = self.get_enforced_action(rb_tick)
                action = self.current_agent.train_with_get_output(state, reward=reward, done=False,
                                                                  enforced_action=enforced_action)
                if not done:
                    self.render_ghost_override()
            else:
                action = self.current_agent.train_with_get_output(state, reward=reward, done=False)

            controller_state = self.get_controller_state_from_actions(action)
            # print(controller_state.__dict__, '2')
        else:
            # controller_state = SimpleControllerState()
            if game_state == GameState.ROUND_WAITING:
                # if current_time - self.current_shot_spawn_time > WAIT_TIME_BEFORE_HIT:
                #     print("gogogo")
                controller_state = SimpleControllerState(throttle=current_time - self.current_shot_spawn_time > WAIT_TIME_BEFORE_HIT)
            else:
                controller_state = SimpleControllerState()

        self.previous_game_state = game_state
        # return None
        # print(controller_state.__dict__)
        return controller_state

    def get_state(self, rb_tick: RigidBodyTick, packet: GameTickPacket, shot_time_elapsed: float) -> np.ndarray:
        ball_state = rb_tick.ball.state
        car_state = rb_tick.players[0].state
        array = np.array([
            car_state.location.x / 4000,
            car_state.location.y / 6000,
            car_state.location.z / 400,
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
            packet.game_cars[0].jumped,

            ball_state.location.x / 4000,
            ball_state.location.y / 6000,
            ball_state.location.z / 400,
            ball_state.velocity.x / 2000,
            ball_state.velocity.y / 2000,
            ball_state.velocity.z / 2000,

            shot_time_elapsed

        ])
        return array

    def get_reward(self, rb_tick: RigidBodyTick, packet: GameTickPacket, done: bool) -> float:
        if done:
            DONE_WEIGHT = 50
            ball_location = rb_tick.ball.state.location
            # print(ball_location)
            # Check if scored
            if 4990 < ball_location.y < 5050 and abs(ball_location.x) < 670 and ball_location.z < 450:
                return DONE_WEIGHT
            else:
                return -DONE_WEIGHT
        reward = 0

        # Get distance from ghost
        ghost_location = self.ghost_handler.get_location(rb_tick)

        car_location = rb_tick.players[0].state.location
        car_location_array = np.array([car_location.x, car_location.y, car_location.z])
        distance_from_ghost = np.sqrt(((car_location_array - ghost_location) ** 2).sum())
        reward += -distance_from_ghost / 10000

        return reward

    def create_agent(self, trained_model_filepaths: Tuple[Path, Path] = None) -> DDPGAgent:
        INPUTS = 21
        OUTPUTS = 8  # steer, throttle, pitch, yaw, roll, jump, boost, handbrake
        if trained_model_filepaths:
            pass
        else:
            actor_model = DenseModel(inputs=INPUTS, outputs=OUTPUTS, layer_nodes=(48, 48), learning_rate=1e-3,
                                     inner_activation='relu', output_activation='tanh')
            critic_model = BaseCriticModel(inputs=INPUTS, outputs=OUTPUTS)
            agent = DDPGAgent(
                actor_model, critic_model=critic_model,
                exploration=OrnsteinUhlenbeck(theta=0.15, sigma=0.05),
            )
            print("Created agent")
            return agent

    @staticmethod
    def get_controller_state_from_actions(action: np.ndarray) -> SimpleControllerState:
        controls = action.clip(-1, 1)
        controller_state = SimpleControllerState()
        controller_state.steer = controls[0]
        controller_state.throttle = controls[1]
        controller_state.pitch = controls[2]
        controller_state.yaw = controls[3]
        controller_state.roll = controls[4]
        controller_state.jump = bool(controls[5] >= 0)
        controller_state.boost = bool(controls[6] >= 0)
        controller_state.handbrake = bool(controls[7] >= 0)
        return controller_state

    @staticmethod
    def get_actions_from_controller_state(controller_state: SimpleControllerState) -> np.ndarray:
        return np.array([
            controller_state.steer,
            controller_state.throttle,
            controller_state.pitch,
            controller_state.yaw,
            controller_state.roll,
            controller_state.jump * 2 - 1,
            controller_state.boost * 2 - 1,
            controller_state.handbrake * 2 - 1,
        ])

    def end_episode_cleanup(self):
        self.total_rewards.append(self.current_shot_total_reward)
        self.episode += 1
        print(f"Completed shot {self.episode}: reward: {self.current_shot_total_reward}")
        self.current_shot_total_reward = 0
        self.ghost_override = random.random() < 0.1
        self.ghost_handler.randomise_current_ghost()

    def get_enforced_action(self, rb_tick: RigidBodyTick) -> np.ndarray:
        ghost_controller_state = self.ghost_handler.get_ghost_controller_state(rb_tick)
        # print(ghost_controller_state.__dict__, 'ghost')
        return self.get_actions_from_controller_state(ghost_controller_state)

    def render_ghost_override(self):
        self.renderer.begin_rendering()
        self.renderer.draw_string_2d(300, 50, 5, 5, "Ghost override", self.renderer.white())
        self.renderer.end_rendering()
