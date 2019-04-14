import time
from typing import Callable, Optional

from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.rendering.rendering_manager import RenderingManager
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.structures.rigid_body_struct import RigidBodyTick

from challenger_bot.base_agent_handler import BaseAgentHandler
from challenger_bot.game_controller import GameState
from challenger_bot.ghosts.ghost import GhostHandler

WAIT_TIME_BEFORE_HIT = 0


class CheeseAgentHandler(BaseAgentHandler):
    def __init__(self, ghost_handler: GhostHandler, renderer: RenderingManager, challenge: str):
        super().__init__(ghost_handler, renderer, challenge)
        self.previous_game_state: Optional[GameState] = None
        self.current_shot_spawn_time: Optional[float] = None

    def get_agent(self, round_: int):
        print("Randomising ghost")
        self.ghost_handler.randomise_current_ghost()

    def challenger_tick(self, packet: GameTickPacket, game_state: GameState,
                        get_rb_tick: Callable[[], RigidBodyTick],
                        previous_packet: GameTickPacket) -> SimpleControllerState:
        if self.current_shot_spawn_time is None or \
                self.previous_game_state != GameState.ROUND_WAITING and game_state == GameState.ROUND_WAITING:
            self.current_shot_spawn_time = time.time()
        current_time = time.time()

        if game_state != GameState.ROUND_ONGOING:
            if game_state == GameState.ROUND_WAITING and \
                    current_time - self.current_shot_spawn_time > WAIT_TIME_BEFORE_HIT:
                controller_state = SimpleControllerState(throttle=1, steer=-0.5, boost=False)
                self.ghost_handler.randomise_current_ghost()
                print("Randomising ghost")
            else:
                controller_state = SimpleControllerState()
        else:
            rb_tick = get_rb_tick()
            controller_state = self.ghost_handler.get_ghost_controller_state(rb_tick)
        # print(controller_state.__dict__)
        self.previous_game_state = game_state

        return controller_state

    def is_setup(self):
        return self.ghost_handler.current_ghost is not None