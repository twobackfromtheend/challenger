import logging

logging.getLogger("tensorflow").setLevel(logging.WARNING)

import os
from pathlib import Path

import math
import numpy as np
import pandas as pd
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from challenger_bot.ds4_interfacer.callbacks import Callback
from challenger_bot.ghosts.ghost import GhostHandler
from challenger_bot.reinforcement_learning.agent_handler import AgentHandler
from challenger_bot.training_pack_data import rounds_ball_data
from challenger_bot.ds4_interfacer.ds4_controller import DS4, DS4Button, DS4Analog

TRAINING_PACK = "7657-2F43-9B3A-C1F1"


class Challenger(BaseAgent):
    # ds4: DS4
    controller_state: SimpleControllerState
    waiting_for_shot: bool
    rounds_ball_data = np.array(rounds_ball_data)
    last_packet: GameTickPacket
    current_round: int = None

    ghost_handler = GhostHandler()
    saving_ghost: bool = False

    ds4_enabled: bool = False

    previous_seconds_elapsed: float = 0

    agent_handler: AgentHandler

    def initialize_agent(self):
        self.logger.info("I LIVE")

        def toggle_ds4_enabled():
            self.ds4_enabled = not self.ds4_enabled
        self.ds4 = DS4(callbacks=[
            Callback(DS4Button.SQUARE, self.setup_round),
            Callback(DS4Button.TOUCHPAD, self.save_ghost_toggle),
            Callback(DS4Button.R_JOYSTICK, toggle_ds4_enabled),
        ])

        self.controller_state = SimpleControllerState()
        self.agent_handler = AgentHandler(self)

        self.waiting_for_shot = True

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # print(packet.game_cars[0].physics.location.x)
        self.ds4.tick()

        current_seconds_elapsed = packet.game_info.seconds_elapsed
        if self.previous_seconds_elapsed == current_seconds_elapsed:
            # Game paused
            self.draw(draw_paused=True)
            return self.controller_state

        round_is_active = packet.game_info.is_round_active
        if round_is_active:
            if self.agent_handler.current_agent is not None:
                self.agent_handler.challenger_tick(packet)

        if self.saving_ghost:
            self.ghost_handler.update(self.get_rigid_body_tick())

        if self.ds4_enabled:
            self.set_controller_state_from_ds4()
            draw_controller_state = False
        else:
            self.controller_state = SimpleControllerState()
            draw_controller_state = False

        self.last_packet = packet
        # print(self.controller_state.__dict__)
        self.previous_seconds_elapsed = current_seconds_elapsed
        self.draw(draw_controller_state=draw_controller_state, draw_ghost=round_is_active)
        return self.controller_state

    def setup_round(self):
        self.detect_round_number()
        self.ghost_handler.get_ghost(self.current_round)
        self.agent_handler.get_agent(self.current_round)

    def detect_round_number(self):
        ball_physics_location = self.last_packet.game_ball.physics.location

        ball_location = np.array([ball_physics_location.x, ball_physics_location.y, ball_physics_location.z])
        ball_distances = np.sqrt((self.rounds_ball_data - ball_location) ** 2).sum(axis=1)

        current_round = ball_distances.argmin() + 1
        self.logger.info(f"Current round of {current_round} (ball dist: {ball_distances[current_round]:.2e})")
        self.current_round = current_round

    def draw(self, draw_paused: bool = False, draw_controller_state: bool = False, draw_ghost: bool = True):
        renderer = self.renderer
        renderer.begin_rendering()

        # Round number
        x_scale = 3
        y_scale = 3
        current_round_str = str(self.current_round) if self.current_round is not None else 'None'
        renderer.draw_rect_2d(10, 95, 500, 50, True, renderer.create_color(80, 0, 0, 0))
        renderer.draw_string_2d(15, 100, x_scale, y_scale, f"ROUND: {current_round_str}",
                                renderer.white())

        # Saving Ghost
        renderer.draw_rect_2d(10, 145, 500, 50, True, renderer.create_color(80, 0, 0, 0))
        renderer.draw_string_2d(15, 150, x_scale, y_scale, f"SAVING GHOST: {self.saving_ghost}",
                                renderer.white())

        # DS4 Enabled
        renderer.draw_rect_2d(10, 195, 500, 50, True, renderer.create_color(80, 0, 0, 0))
        renderer.draw_string_2d(15, 200, x_scale, y_scale, f"DS4 Enabled: {self.ds4_enabled}",
                                renderer.white())

        # Draw ghost
        if draw_ghost:
            ghost_location = self.ghost_handler.get_location(self.current_round, self.get_rigid_body_tick())
            renderer.draw_rect_3d(ghost_location, 20, 20, True, renderer.white())

        if draw_paused:
            renderer.draw_rect_2d(10, 395, 500, 135, True, renderer.create_color(80, 0, 0, 0))
            renderer.draw_string_2d(15, 400, 8, 8, f"PAUSED", renderer.red())

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
            self.saving_ghost = True
