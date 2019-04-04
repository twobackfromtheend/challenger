import os
from pathlib import Path

import math
import numpy as np
import pandas as pd
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from challenger_bot.ds4_interfacer.callbacks import Callback
from challenger_bot.ghosts.ghost import GhostHandler
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

    def initialize_agent(self):
        self.logger.info("I LIVE")

        self.ds4 = DS4(callbacks=[
            Callback(DS4Button.SQUARE, self.detect_round_number),
            Callback(DS4Button.TOUCHPAD, self.save_ghost_toggle),
        ])

        self.controller_state = SimpleControllerState()

        self.waiting_for_shot = True

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        self.ds4.tick()
        ball_physics = packet.game_ball.physics
        # if ball_physics.velocity.x == 0 and ball_physics.velocity.y == 0 and ball_physics.velocity.z == 0:
        #     self.detect_round_number(packet)

        if self.saving_ghost:
            self.ghost_handler.update(self.get_rigid_body_tick())

        self.draw()

        self.set_controller_state_from_ds4()
        self.last_packet = packet
        # print(self.controller_state.__dict__)
        return self.controller_state

    def detect_round_number(self):
        ball_physics_location = self.last_packet.game_ball.physics.location

        ball_location = np.array([ball_physics_location.x, ball_physics_location.y, ball_physics_location.z])
        ball_distances = np.sqrt((self.rounds_ball_data - ball_location)**2).sum(axis=1)

        current_round = ball_distances.argmin() + 1
        self.logger.info(f"Current round of {current_round} (ball dist: {ball_distances[current_round]:.2e})")
        self.current_round = current_round

    def draw(self):
        renderer = self.renderer
        renderer.begin_rendering()

        # Round number
        x_scale = 3
        y_scale = 4
        current_round_str = str(self.current_round) if self.current_round is not None else 'None'
        renderer.draw_string_2d(15, 100, x_scale, y_scale, f"ROUND: {current_round_str}",
                                renderer.white())

        # Saving Ghost
        renderer.draw_string_2d(15, 150, x_scale, y_scale, f"SAVING GHOST: {self.saving_ghost}",
                                renderer.white())

        # Draw ghost
        ghost_location = self.ghost_handler.get_location(self.current_round, self.get_rigid_body_tick())
        renderer.draw_rect_3d(ghost_location, 20, 20, True, renderer.white())

        renderer.end_rendering()

    def set_controller_state_from_ds4(self):
        self.controller_state.boost = self.ds4.get_button(DS4Button.O)
        self.controller_state.jump = self.ds4.get_button(DS4Button.X)
        l_horizontal = self.apply_deadzone_center(self.ds4.get_button(DS4Analog.L_HORIZONTAL))

        self.controller_state.throttle = (self.ds4.get_button(DS4Analog.R2) - self.ds4.get_button(DS4Analog.L2)) / 2
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
        DEADZONE = 0.2
        value = max(value + 1 - DEADZONE, 0) / (2 - DEADZONE) - 1
        return value

    def save_ghost_toggle(self):
        if self.saving_ghost:
            self.saving_ghost = False
            self.ghost_handler.save_ghost(self.current_round)
        else:
            self.saving_ghost = True
