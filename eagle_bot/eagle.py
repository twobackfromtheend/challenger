import copy
import logging
import random
import time

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import math
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState, Vector3, Rotator, BallState, Physics, CarState

from eagle_bot.ds4_interfacer.callbacks import Callback
from eagle_bot.reinforcement_learning.agent_handler import AgentHandler
from eagle_bot.ds4_interfacer.ds4_controller import DS4, DS4Button


class EagleBot(BaseAgent):
    ds4: DS4
    controller_state: SimpleControllerState
    previous_packet: GameTickPacket = None
    current_round: int = None

    ds4_enabled: bool = False

    agent_handler: AgentHandler

    has_reset: bool = False
    has_started_episode: bool = False
    episode_start_time: float = time.time()

    def initialize_agent(self):
        self.ds4 = DS4(callbacks=[
            Callback(DS4Button.SQUARE, self.reset_shot),
            # Callback(DS4Button.L_JOYSTICK, game_controller.reset_shot),
        ])

        self.controller_state = SimpleControllerState()
        self.agent_handler = AgentHandler(renderer=self.renderer)
        self.agent_handler.get_agent()
        self.logger.info("I LIVE")

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        if self.previous_packet is None:
            self.previous_packet = packet
        self.ds4.tick()

        current_seconds_elapsed = packet.game_info.seconds_elapsed
        previous_seconds_elapsed = self.previous_packet.game_info.seconds_elapsed
        # print(current_seconds_elapsed, previous_seconds_elapsed, previous_seconds_elapsed == current_seconds_elapsed)
        if previous_seconds_elapsed == current_seconds_elapsed:
            # PAUSED
            # self.draw(game_state=current_game_state)
            self.previous_packet = copy.copy(packet)
            return self.controller_state

        current_time = time.time()
        if self.has_started_episode:
            if current_time - self.episode_start_time > 0.05:
                # Get controls based on car_controller
                self.controller_state, done = self.agent_handler.challenger_tick(packet, self.get_rigid_body_tick,
                                                                                 self.previous_packet)

                # Game state control
                current_time = time.time()
                if done or current_time - self.episode_start_time > 60:
                    self.reset_shot()

        else:
            self.episode_start_time = current_time
            # self.reset_shot()
            self.has_started_episode = True

        self.previous_packet = copy.copy(packet)
        # print(self.controller_state.__dict__)

        # self.draw(, draw_controller_state=draw_controller_state)
        # print(any(list(self.controller_state.__dict__.values())))
        return self.controller_state

    def reset_shot(self):
        zero_vector = Vector3(x=0, y=0, z=0)

        ball_start_location = Vector3(x=0, y=0, z=800)

        car_start_location = Vector3(x=0, y=0, z=600)
        car_start_rotation = Rotator(
            # pitch=math.pi / 2,
            pitch=math.pi / 2 + random.random() * 0.1 * math.pi,
            yaw=random.random() * 2 * math.pi,
            roll=random.random() * 2 * math.pi
        )

        game_state = GameState(
            ball=BallState(Physics(location=ball_start_location, velocity=zero_vector, angular_velocity=zero_vector)),
            cars={self.index: CarState(
                physics=Physics(location=car_start_location,
                                rotation=car_start_rotation,
                                velocity=zero_vector,
                                angular_velocity=zero_vector
                                )
            )}
        )
        self.set_game_state(game_state)
        self.has_started_episode = False
        self.has_reset = True
        print('hi')

    # def draw(self, game_state: GameState, draw_controller_state: bool = False):
    #     renderer = self.renderer
    #     renderer.begin_rendering()
    #
    #     # Round number
    #     x_scale = 3
    #     y_scale = 3
    #     y_offset = 50
    #     current_round_str = str(self.current_round) if self.current_round is not None else 'None'
    #     renderer.draw_rect_2d(10, 95 + y_offset, 500, 50, True, renderer.create_color(80, 0, 0, 0))
    #     renderer.draw_string_2d(15, 100 + y_offset, x_scale, y_scale, f"ROUND: {current_round_str}",
    #                             renderer.white())
    #
    #     # if game_state == GameState.PAUSED:
    #     #     renderer.draw_rect_2d(10, 395, 500, 135, True, renderer.create_color(80, 0, 0, 0))
    #     #     renderer.draw_string_2d(15, 400, 8, 8, f"PAUSED", renderer.red())
    #     game_state_draw_data = {
    #         GameState.PAUSED: {
    #             'color': renderer.red(),
    #             'scale': 8,
    #             'width': 430,
    #             'height': 135,
    #         },
    #         GameState.ROUND_WAITING: {
    #             'color': renderer.orange(),
    #             'scale': 4,
    #             'width': 460,
    #             'height': 80,
    #         },
    #         GameState.ROUND_ONGOING: {
    #             'color': renderer.lime(),
    #             'scale': 4,
    #             'width': 470,
    #             'height': 80,
    #         },
    #         GameState.ROUND_FINISHED: {
    #             'color': renderer.grey(),
    #             'scale': 4,
    #             'width': 470,
    #             'height': 80,
    #         },
    #         GameState.REPLAY: {
    #             'color': renderer.white() if int(time.time() * 3) % 2 == 0 else renderer.grey(),
    #             'scale': 4,
    #             'width': 210,
    #             'height': 80,
    #         },
    #     }
    #     draw_data = game_state_draw_data[game_state]
    #     renderer.draw_rect_2d(10, 395, draw_data['width'], draw_data['height'], True,
    #                           renderer.create_color(80, 0, 0, 0))
    #     renderer.draw_string_2d(15, 405, draw_data['scale'], draw_data['scale'], f"{game_state.name}",
    #                             draw_data['color'])
    #
    #     if draw_controller_state:
    #         x_scale = 1
    #         y_scale = 1
    #         renderer = self.renderer
    #         renderer.begin_rendering()
    #         x = 500
    #         y = 100
    #         for control in ["steer", "throttle", "pitch", "yaw", "roll"]:
    #             renderer.draw_string_2d(x, y, x_scale, y_scale, f"{control}: {getattr(self.controller_state, control)}",
    #                                     renderer.white())
    #             y += 10
    #     renderer.end_rendering()
    #
    # def set_controller_state_from_ds4(self):
    #     self.controller_state.boost = self.ds4.get_button(DS4Button.O)
    #     self.controller_state.jump = self.ds4.get_button(DS4Button.X)
    #     l_horizontal = self.apply_deadzone_center(self.ds4.get_button(DS4Analog.L_HORIZONTAL))
    #
    #     r2_deadzoned = self.apply_deadzone_small(self.ds4.get_button(DS4Analog.R2))
    #     l1_deadzoned = self.apply_deadzone_small(self.ds4.get_button(DS4Analog.L2))
    #     self.controller_state.throttle = (r2_deadzoned - l1_deadzoned) / 2
    #     # self.logger.info(self.ds4.get_button(DS4Analog.R2), self.ds4.get_button(DS4Analog.L2))
    #     self.controller_state.steer = l_horizontal
    #     self.controller_state.pitch = self.apply_deadzone_center(self.ds4.get_button(DS4Analog.L_VERTICAL))
    #     if self.ds4.get_button(DS4Button.L1):
    #         self.controller_state.roll = l_horizontal
    #         self.controller_state.yaw = 0
    #     else:
    #         self.controller_state.yaw = l_horizontal
    #         self.controller_state.roll = 0
    #     # print(self.ds4.get_button(DS4Button.L1), self.controller_state.roll)
    #     # print(self.controller_state.__dict__)
    #     for control in ['steer', 'throttle', 'pitch', 'yaw', 'roll']:
    #         setattr(self.controller_state, control, max(min(getattr(self.controller_state, control), 1), -1))
    #
    # @staticmethod
    # def apply_deadzone_center(value: float):
    #     DEADZONE = 0.1
    #     adjusted_magnitude = max(abs(value) - DEADZONE, 0)
    #     return math.copysign(adjusted_magnitude, value) / (1 - DEADZONE)
    #
    # @staticmethod
    # def apply_deadzone_small(value: float):
    #     DEADZONE = 0.05
    #     value = max(value + 1 - DEADZONE, 0) * 2. / (2 - DEADZONE) - 1
    #     return min(max(value * 1 / 0.88, -1), 1)  # Adjust for controller range (-1 to 0.88)
