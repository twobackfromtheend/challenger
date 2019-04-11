import copy
import logging
import time
from typing import Union, Optional

from challenger_bot.car_controllers import CarController
from challenger_bot.game_controller import GameState

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import math
import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from challenger_bot import game_controller
from challenger_bot.cheese.cheese_agent_handler import CheeseAgentHandler
from challenger_bot.ds4_interfacer.callbacks import Callback
from challenger_bot.ghosts.ghost import GhostHandler
from challenger_bot.reinforcement_learning.agent_handler import AgentHandler
from challenger_bot.training_pack_data import rounds_ball_data
from challenger_bot.ds4_interfacer.ds4_controller import DS4, DS4Button, DS4Analog

TRAINING_PACK = "7657-2F43-9B3A-C1F1"
CHALLENGE = TRAINING_PACK


class Challenger(BaseAgent):
    challenge: str = CHALLENGE

    ds4: DS4
    controller_state: SimpleControllerState
    waiting_for_shot: bool
    rounds_ball_data = np.array(rounds_ball_data)
    previous_packet: GameTickPacket = None
    current_round: int = None

    game_state_finished_time: Optional[float] = None

    ghost_handler = GhostHandler(CHALLENGE)
    saving_ghost: bool = False

    ds4_enabled: bool = False

    agent_handler: Union[AgentHandler, CheeseAgentHandler]

    current_car_controller: CarController = CarController.AGENT

    def initialize_agent(self):
        self.logger.info("I LIVE")

        def toggle_car_controller():
            if self.current_car_controller == CarController.AGENT:
                self.current_car_controller = CarController.DS4
            else:
                self.current_car_controller = CarController.AGENT

        self.ds4 = DS4(callbacks=[
            Callback(DS4Button.SQUARE, self.setup_round),
            Callback(DS4Button.TOUCHPAD, self.save_ghost_toggle),
            Callback(DS4Button.R_JOYSTICK, toggle_car_controller),
            # Callback(DS4Button.L_JOYSTICK, game_controller.reset_shot),
        ])

        self.controller_state = SimpleControllerState()
        self.agent_handler = AgentHandler(ghost_handler=self.ghost_handler, renderer=self.renderer,
                                          challenge=self.challenge)

        self.waiting_for_shot = True

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        if self.previous_packet is None:
            self.previous_packet = packet
        # print(packet.game_cars[0].physics.location.x)
        self.ds4.tick()
        current_game_state: GameState = game_controller.get_game_state(packet, self.previous_packet)

        # Setup round when waiting
        # if current_game_state == GameState.ROUND_WAITING:
        #     self.setup_round()

        # Do nothing if paused
        if current_game_state == GameState.PAUSED:
            self.draw(game_state=current_game_state)
            self.previous_packet = copy.copy(packet)
            return self.controller_state

        # Get controls based on car_controller
        if self.current_car_controller == CarController.AGENT and self.agent_handler.is_setup():
            self.controller_state = self.agent_handler.challenger_tick(packet, current_game_state,
                                                                       self.get_rigid_body_tick)
            # print(self.controller_state.__dict__)
            draw_controller_state = False
        else:
            if self.saving_ghost:
                self.ghost_handler.update(self.get_rigid_body_tick())

            if self.current_car_controller == CarController.DS4:
                self.set_controller_state_from_ds4()
                draw_controller_state = False
            else:
                self.controller_state = SimpleControllerState()
                draw_controller_state = False

        # Game state control
        if current_game_state == GameState.REPLAY:
            game_controller.reset_shot()

        if current_game_state == GameState.ROUND_FINISHED:
            if self.game_state_finished_time is None:
                self.game_state_finished_time = time.time()
            if time.time() - self.game_state_finished_time > 1:
                game_controller.reset_shot()
                self.game_state_finished_time = None

        self.previous_packet = copy.copy(packet)
        # print(self.controller_state.__dict__)
        self.draw(game_state=current_game_state, draw_controller_state=draw_controller_state)
        # print(any(list(self.controller_state.__dict__.values())))
        return self.controller_state

    def setup_round(self):
        self.detect_round_number()
        self.ghost_handler.get_ghosts(self.current_round)
        self.agent_handler.get_agent(self.current_round)

    def detect_round_number(self):
        ball_physics_location = self.previous_packet.game_ball.physics.location

        ball_location = np.array([ball_physics_location.x, ball_physics_location.y, ball_physics_location.z])
        ball_distances = np.sqrt((self.rounds_ball_data - ball_location) ** 2).sum(axis=1)

        current_round = ball_distances.argmin() + 1
        # self.logger.info(f"Current round of {current_round} (ball dist: {ball_distances[current_round]:.2e})")
        self.current_round = current_round

    def draw(self, game_state: GameState, draw_controller_state: bool = False):
        renderer = self.renderer
        renderer.begin_rendering()

        # Round number
        x_scale = 3
        y_scale = 3
        y_offset = 50
        current_round_str = str(self.current_round) if self.current_round is not None else 'None'
        renderer.draw_rect_2d(10, 95 + y_offset, 500, 50, True, renderer.create_color(80, 0, 0, 0))
        renderer.draw_string_2d(15, 100 + y_offset, x_scale, y_scale, f"ROUND: {current_round_str}",
                                renderer.white())

        # Saving Ghost
        renderer.draw_rect_2d(10, 145 + y_offset, 500, 50, True, renderer.create_color(80, 0, 0, 0))
        renderer.draw_string_2d(15, 150 + y_offset, x_scale, y_scale, f"SAVING GHOST: {self.saving_ghost}",
                                renderer.white())

        # Car controller
        renderer.draw_rect_2d(10, 195 + y_offset, 500, 50, True, renderer.create_color(80, 0, 0, 0))
        renderer.draw_string_2d(15, 200 + y_offset, x_scale, y_scale, f"Controller: {self.current_car_controller.name}",
                                renderer.white())

        # # Agent Handler Enabled
        # renderer.draw_rect_2d(10, 245, 500, 50, True, renderer.create_color(80, 0, 0, 0))
        # renderer.draw_string_2d(15, 250, x_scale, y_scale, f"Agent Enabled: {self.agent_handler_enabled}",
        #                         renderer.white())

        if self.current_car_controller == CarController.AGENT and isinstance(self.agent_handler, CheeseAgentHandler):
            renderer.draw_string_2d(200, 500, 10, 10, "CHEESING", renderer.yellow())

        # Draw ghost
        try:
            ghost_location = self.ghost_handler.get_location(self.get_rigid_body_tick())
            renderer.draw_rect_3d(ghost_location, 20, 20, True, renderer.white())
        except:
            pass
        # if game_state == GameState.PAUSED:
        #     renderer.draw_rect_2d(10, 395, 500, 135, True, renderer.create_color(80, 0, 0, 0))
        #     renderer.draw_string_2d(15, 400, 8, 8, f"PAUSED", renderer.red())
        game_state_draw_data = {
            GameState.PAUSED: {
                'color': renderer.red(),
                'scale': 8,
                'width': 430,
                'height': 135,
            },
            GameState.ROUND_WAITING: {
                'color': renderer.orange(),
                'scale': 4,
                'width': 460,
                'height': 80,
            },
            GameState.ROUND_ONGOING: {
                'color': renderer.lime(),
                'scale': 4,
                'width': 470,
                'height': 80,
            },
            GameState.ROUND_FINISHED: {
                'color': renderer.grey(),
                'scale': 4,
                'width': 470,
                'height': 80,
            },
            GameState.REPLAY: {
                'color': renderer.white() if int(time.time() * 3) % 2 == 0 else renderer.grey(),
                'scale': 4,
                'width': 210,
                'height': 80,
            },
        }
        draw_data = game_state_draw_data[game_state]
        renderer.draw_rect_2d(10, 395, draw_data['width'], draw_data['height'], True,
                              renderer.create_color(80, 0, 0, 0))
        renderer.draw_string_2d(15, 405, draw_data['scale'], draw_data['scale'], f"{game_state.name}",
                                draw_data['color'])

        if draw_controller_state:
            x_scale = 1
            y_scale = 1
            renderer = self.renderer
            renderer.begin_rendering()
            x = 500
            y = 100
            for control in ["steer", "throttle", "pitch", "yaw", "roll"]:
                renderer.draw_string_2d(x, y, x_scale, y_scale, f"{control}: {getattr(self.controller_state, control)}",
                                        renderer.white())
                y += 10
        renderer.end_rendering()

    def set_controller_state_from_ds4(self):
        self.controller_state.boost = self.ds4.get_button(DS4Button.O)
        self.controller_state.jump = self.ds4.get_button(DS4Button.X)
        l_horizontal = self.apply_deadzone_center(self.ds4.get_button(DS4Analog.L_HORIZONTAL))

        r2_deadzoned = self.apply_deadzone_small(self.ds4.get_button(DS4Analog.R2))
        l1_deadzoned = self.apply_deadzone_small(self.ds4.get_button(DS4Analog.L2))
        self.controller_state.throttle = (r2_deadzoned - l1_deadzoned) / 2
        # self.logger.info(self.ds4.get_button(DS4Analog.R2), self.ds4.get_button(DS4Analog.L2))
        self.controller_state.steer = l_horizontal
        self.controller_state.pitch = self.apply_deadzone_center(self.ds4.get_button(DS4Analog.L_VERTICAL))
        if self.ds4.get_button(DS4Button.L1):
            self.controller_state.roll = l_horizontal
            self.controller_state.yaw = 0
        else:
            self.controller_state.yaw = l_horizontal
            self.controller_state.roll = 0
        # print(self.ds4.get_button(DS4Button.L1), self.controller_state.roll)
        # print(self.controller_state.__dict__)
        for control in ['steer', 'throttle', 'pitch', 'yaw', 'roll']:
            setattr(self.controller_state, control, max(min(getattr(self.controller_state, control), 1), -1))

    @staticmethod
    def apply_deadzone_center(value: float):
        DEADZONE = 0.1
        adjusted_magnitude = max(abs(value) - DEADZONE, 0)
        return math.copysign(adjusted_magnitude, value) / (1 - DEADZONE)

    @staticmethod
    def apply_deadzone_small(value: float):
        DEADZONE = 0.05
        value = max(value + 1 - DEADZONE, 0) * 2. / (2 - DEADZONE) - 1
        return min(max(value * 1 / 0.88, -1), 1)  # Adjust for controller range (-1 to 0.88)

    def save_ghost_toggle(self):
        if self.saving_ghost:
            self.saving_ghost = False
            self.ghost_handler.save_ghost(self.current_round)
        else:
            if self.current_car_controller == CarController.DS4:
                self.saving_ghost = True
