from typing import Callable

from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.rendering.rendering_manager import RenderingManager
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.structures.rigid_body_struct import RigidBodyTick

from challenger_bot.game_controller import GameState
from challenger_bot.ghosts.ghost import GhostHandler


class BaseAgentHandler:
    def __init__(self, ghost_handler: GhostHandler, renderer: RenderingManager, challenge: str):
        self.ghost_handler = ghost_handler
        self.renderer = renderer
        self.challenge = challenge

    def is_setup(self) -> bool:
        raise NotImplementedError

    def challenger_tick(self, packet: GameTickPacket, game_state: GameState,
                        get_rb_tick: Callable[[], RigidBodyTick],
                        previous_packet: GameTickPacket) -> SimpleControllerState:
        raise NotImplementedError
