from typing import Callable

from rlbot.agents.base_agent import SimpleControllerState
from rlbot.messages.flat import GameTickPacket
from rlbot.utils.rendering.rendering_manager import RenderingManager
from rlbot.utils.structures.rigid_body_struct import RigidBodyTick

from challenger_bot.base_agent_handler import BaseAgentHandler
from challenger_bot.game_controller import GameState
from challenger_bot.ghosts.ghost import GhostHandler


class CheeseAgentHandler(BaseAgentHandler):
    def __init__(self, ghost_handler: GhostHandler, renderer: RenderingManager):
        super().__init__(ghost_handler, renderer)

    def get_agent(self, round: int):
        pass

    def challenger_tick(self, packet: GameTickPacket, game_state: GameState,
                        get_rb_tick: Callable[[], RigidBodyTick]) -> SimpleControllerState:
        if game_state != GameState.ROUND_ONGOING:
            return SimpleControllerState()

        rb_tick = get_rb_tick()
        controller_state = self.ghost_handler.get_ghost_controller_state(rb_tick)

        # return None
        return controller_state



    def is_setup(self):
        return self.ghost_handler.current_ghost is not None