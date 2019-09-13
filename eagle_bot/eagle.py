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
                if done:
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
