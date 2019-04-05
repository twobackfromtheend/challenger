from typing import TYPE_CHECKING

from rlbot.agents.base_agent import SimpleControllerState
from rlbot.messages.flat import GameTickPacket

from challenger_bot.base_agent_handler import BaseAgentHandler
from challenger_bot.game_controller import GameState

if TYPE_CHECKING:
    from challenger_bot.challenger import Challenger


class CheeseAgentHandler(BaseAgentHandler):
    def __init__(self, challenger: 'Challenger'):
        self.challenger = challenger
        self.ghost_handler = self.challenger.ghost_handler

        self.previous_round_is_active: bool = False

    def get_agent(self, round: int):
        pass

    def challenger_tick(self, packet: GameTickPacket, game_state: GameState) -> SimpleControllerState:
        round_is_active = packet.game_info.is_round_active
        if not round_is_active and not self.previous_round_is_active:
            return SimpleControllerState()

        rb_tick = self.challenger.get_rigid_body_tick()

        current_frame_delta = self.ghost_handler.get_current_frame_delta(rb_tick)
        ghost = self.ghost_handler.current_ghost
        controls = ghost.loc[
            current_frame_delta,
            ['steer', 'throttle', 'pitch', 'yaw', 'roll', 'boost', 'jump', 'handbrake']
        ]
        controls = controls.clip(-1, 1)
        controller_state = SimpleControllerState()
        controller_state.steer, controller_state.throttle, \
            controller_state.pitch, controller_state.yaw, controller_state.roll, \
            controller_state.boost, controller_state.jump, controller_state.handbrake = controls

        self.previous_round_is_active = round_is_active
        # return None
        return controller_state

    def is_setup(self):
        return self.ghost_handler.current_ghost is not None