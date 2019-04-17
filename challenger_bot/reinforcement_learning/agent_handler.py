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
from challenger_bot.reinforcement_learning.model_handler import get_model_savepath, get_latest_model

if TYPE_CHECKING:
    from challenger_bot.ghosts.ghost import GhostHandler
    from rlbot.utils.rendering.rendering_manager import RenderingManager

WAIT_TIME_BEFORE_HIT = 0


class AgentHandler(BaseAgentHandler):
    trained_models_folder = Path(__file__).parent / "trained_models"

    def __init__(self, ghost_handler: 'GhostHandler', renderer: 'RenderingManager', challenge: str):
        super().__init__(ghost_handler, renderer, challenge)
        self.challenge_trained_models_folder: Path = self.trained_models_folder / challenge
        self.round_trained_models_folder: Optional[Path] = None

        self.current_agent: Optional[DDPGAgent] = None

        self.training_rewards = []
        self.episode = 0
        self.evaluation_rewards = []
        self.evaluations: int = 0

        self.goals = 0
        self.ghost_goals = 0
        self.evaluation_goals: int = 0

        self.ghost_override: bool = False
        self.evaluation: bool = False

        self.previous_game_state: Optional[GameState] = None
        self.current_shot_spawn_time: Optional[float] = None
        self.current_shot_total_reward = 0
        self.current_shot_start_time: Optional[float] = None

        self.has_touched = False
        self.latest_touch_time = 0
        self.last_controller_state = None
        self.just_jumped = False
        self.has_touched_ceiling = False

    def get_agent(self, round_: int):
        self.round_trained_models_folder: Path = self.challenge_trained_models_folder / str(round_)
        self.round_trained_models_folder.mkdir(exist_ok=True)

        # DESIRED_MODEL = "_20190411-113911.h5"
        DESIRED_MODEL = ""
        trained_actor_filepath = get_latest_model(self.round_trained_models_folder, "actor*" + DESIRED_MODEL)
        trained_critic_filepath = get_latest_model(self.round_trained_models_folder, "critic*" + DESIRED_MODEL)

        if trained_actor_filepath is not None and trained_critic_filepath is not None and \
                trained_actor_filepath.is_file() and trained_critic_filepath.is_file():
            self.current_agent = self.create_agent((trained_actor_filepath, trained_critic_filepath))
        else:
            self.current_agent = self.create_agent()

    def is_setup(self) -> bool:
        return self.current_agent is not None

    def challenger_tick(self, packet: GameTickPacket, game_state: GameState,
                        get_rb_tick: Callable[[], RigidBodyTick],
                        previous_packet: GameTickPacket) -> SimpleControllerState:
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

            if self.evaluation:
                action = self.current_agent.train_with_get_output(state, reward=reward, done=False, evaluation=True)
            elif self.ghost_override:
                enforced_action = self.get_enforced_action(rb_tick)
                action = self.current_agent.train_with_get_output(state, reward=reward, done=False,
                                                                  enforced_action=enforced_action)
            else:
                action = self.current_agent.train_with_get_output(state, reward=reward, done=False)

            controller_state = self.get_controller_state_from_actions(action)
            if self.last_controller_state is not None and not self.last_controller_state.jump and controller_state.jump:
                self.just_jumped = True
            else:
                self.just_jumped = False
            self.last_controller_state = controller_state
            # print(controller_state.__dict__, '2')
        else:
            # controller_state = SimpleControllerState()
            if game_state == GameState.ROUND_WAITING:
                # if current_time - self.current_shot_spawn_time > WAIT_TIME_BEFORE_HIT:
                #     print("gogogo")
                controller_state = SimpleControllerState(
                    throttle=current_time - self.current_shot_spawn_time > WAIT_TIME_BEFORE_HIT)
            else:
                controller_state = SimpleControllerState()

        self.draw(game_state)

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
        touched_this_frame = False
        latest_touch = packet.game_ball.latest_touch
        if latest_touch.time_seconds != self.latest_touch_time:
            self.latest_touch_time = latest_touch.time_seconds
            touched_this_frame = True
        if done:
            DONE_WEIGHT = 30
            ball_location = rb_tick.ball.state.location
            ball_distance_from_goal = ((ball_location.y - 5010) ** 2 + abs(ball_location.x) ** 2 + (
                        ball_location.z - 300) ** 2) ** 0.5
            reward = 5 - ball_distance_from_goal / 1000
            # print(ball_distance_from_goal, reward)
            # print(f"ball distance from goal reward: {reward}")
            # print(ball_location)
            # Check if scored
            if 4995 < ball_location.y < 5050 and abs(ball_location.x) < 900 and ball_location.z < 650:
                if self.evaluation:
                    self.evaluation_goals += 1
                else:
                    self.goals += 1
                    if self.ghost_override:
                        self.ghost_goals += 1
                return DONE_WEIGHT
            else:
                return -DONE_WEIGHT + reward

        reward = 0

        car_location = rb_tick.players[0].state.location
        # Reward for flip reset off ceiling in specified place
        # if not self.has_touched_ceiling and\
        #         -1380 < car_location.x < -1280 and -2550 < car_location.y < -2450 and\
        if not self.has_touched_ceiling and \
                car_location.z > 2020 and -1700 < car_location.x < -900 and -3000 < car_location.y < -2000:
            if packet.game_cars[0].has_wheel_contact:
                reward += 20
                print("Touched ceiling")
                self.has_touched_ceiling = True

        # Punishment for boosting / jumping
        if self.last_controller_state is not None:
            # if self.last_controller_state.boost:
            #     reward -= 0.04
            if self.just_jumped:
                reward -= 5

        # Reward for first touch
        if touched_this_frame and not self.has_touched and self.has_touched_ceiling:
            reward += 40
            print("Touched ball")
            self.has_touched = True

        # Get distance from ghost
        ghost_location = self.ghost_handler.get_location(rb_tick)

        # car_location = rb_tick.players[0].state.location
        car_location_array = np.array([car_location.x, car_location.y, car_location.z])
        distance_from_ghost = np.sqrt(((car_location_array - ghost_location) ** 2).sum())
        reward += -distance_from_ghost / 10000

        return reward

    def create_agent(self, trained_model_filepaths: Tuple[Path, Path] = None) -> DDPGAgent:
        INPUTS = 21
        OUTPUTS = 8  # steer, throttle, pitch, yaw, roll, jump, boost, handbrake
        if trained_model_filepaths:
            actor_model = DenseModel(INPUTS, OUTPUTS, load_from_filepath=str(trained_model_filepaths[0]))
            critic_model = BaseCriticModel(INPUTS, OUTPUTS,
                                           load_from_filepath=str(trained_model_filepaths[1]))
            print(f"Loaded trained actor: {trained_model_filepaths[0].name} and "
                  f"trained critic: {trained_model_filepaths[1].name}")
        else:
            actor_model = DenseModel(inputs=INPUTS, outputs=OUTPUTS, layer_nodes=(128, 128, 128),
                                     inner_activation='relu', output_activation='tanh')
            critic_model = BaseCriticModel(inputs=INPUTS, outputs=OUTPUTS)
        agent = DDPGAgent(
            actor_model, critic_model=critic_model,
            exploration=OrnsteinUhlenbeck(theta=0.15, sigma=0.03, dt=1 / 60, size=OUTPUTS),
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
        self.has_touched = False
        self.just_jumped = False
        self.has_touched_ceiling = False
        self.current_shot_start_time = None

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
            self.ghost_override = False
            if self.episode % 200 == 1 and self.episode > 500:
                print(f"Saving models: {self.round_trained_models_folder}")
                actor_savepath = get_model_savepath(self.round_trained_models_folder, f"actor_{self.episode}")
                critic_savepath = get_model_savepath(self.round_trained_models_folder, f"critic_{self.episode}")
                self.current_agent.actor_model.save_model(actor_savepath)
                self.current_agent.critic_model.save_model(critic_savepath)
        else:
            self.evaluation = False
            self.ghost_override = random.random() < 0.3
            self.ghost_handler.randomise_current_ghost()

    def get_enforced_action(self, rb_tick: RigidBodyTick) -> np.ndarray:
        ghost_controller_state = self.ghost_handler.get_ghost_controller_state(rb_tick)
        # print(ghost_controller_state.__dict__, 'ghost')
        return self.get_actions_from_controller_state(ghost_controller_state)

    def draw(self, game_state: GameState):
        renderer = self.renderer
        renderer.begin_rendering('agent_handler')
        x_scale = y_scale = 3
        renderer.draw_rect_2d(1095, 95, 440, 50, True, renderer.create_color(80, 0, 0, 0))
        renderer.draw_string_2d(1100, 100, x_scale, y_scale, f"EPISODE: {self.episode}",
                                renderer.white())

        renderer.draw_rect_2d(1095, 145, 440, 50, True, renderer.create_color(80, 0, 0, 0))
        renderer.draw_string_2d(1100, 150, x_scale, y_scale,
                                f"GOALS: {self.goals} ({self.goals - self.ghost_goals} ML)",
                                renderer.white())

        renderer.draw_rect_2d(1095, 195, 440, 50, True, renderer.create_color(80, 0, 0, 0))
        renderer.draw_string_2d(1100, 200, x_scale, y_scale,
                                f"EVAL: {self.evaluation_goals} / {self.evaluations}",
                                renderer.white())

        if game_state == GameState.ROUND_WAITING or game_state == GameState.ROUND_ONGOING:
            if self.evaluation:
                renderer.draw_string_2d(1050, 20, 5, 5, "EVALUATING",
                                        renderer.white() if int(time.time() * 3) % 2 == 0 else renderer.grey())
            elif self.ghost_override:
                renderer.draw_string_2d(950, 20, 5, 5, "GHOST OVERRIDE",
                                        renderer.white() if int(time.time() * 3) % 2 == 0 else renderer.grey())

        renderer.end_rendering()
