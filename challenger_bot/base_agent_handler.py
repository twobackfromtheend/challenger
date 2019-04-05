from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from challenger_bot.game_controller import GameState


class BaseAgentHandler:
    def is_setup(self) -> bool:
        raise NotImplementedError

    def challenger_tick(self, packet: GameTickPacket, game_state: GameState) -> SimpleControllerState:
        raise NotImplementedError
