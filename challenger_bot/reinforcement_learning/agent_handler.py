from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from rlbot.agents.base_agent import SimpleControllerState
from rlbot.messages.flat import GameTickPacket
from rlbot.utils.structures.rigid_body_struct import RigidBodyTick

from challenger_bot.reinforcement_learning.agent import DDPGAgent
from challenger_bot.reinforcement_learning.exploration.ornstein_uhlenbeck import OrnsteinUhlenbeck
from challenger_bot.reinforcement_learning.model.base_critic_model import BaseCriticModel
from challenger_bot.reinforcement_learning.model.dense_model import DenseModel

if TYPE_CHECKING:
    from challenger_bot.challenger import Challenger

trained_models_folder = Path(__file__).parent / "trained_models"


class AgentHandler:
    def __init__(self, challenger: 'Challenger'):
        self.trained_actor_filepath: Optional[Path] = None
        self.trained_critic_filepath: Optional[Path] = None

        self.challenger = challenger
        self.current_agent: Optional[DDPGAgent] = None
        self.previous_round_is_active: bool = False

    def get_agent(self, round: int):
        trained_actor_filepath = trained_models_folder / f"{round}_actor.h5"
        trained_critic_filepath = trained_models_folder / f"{round}_critic.h5"
        if trained_actor_filepath.is_file() and trained_critic_filepath.is_file():
            self.current_agent = self.create_agent((trained_actor_filepath, trained_critic_filepath))
        else:
            self.current_agent = self.create_agent()

    def challenger_tick(self, packet: GameTickPacket):
        if self.current_agent is None:
            raise Exception("Not loaded agent.")

        round_is_active = packet.game_info.is_round_active
        if not round_is_active:
            if self.previous_round_is_active:
                # Done step
                done = True
            else:
                return
                # TODO: Round finished. stop training
        else:
            done = False

        rb_tick = self.challenger.get_rigid_body_tick()
        state = self.get_state(rb_tick, packet)
        reward = self.get_reward(rb_tick, packet, done)
        print(reward)

        action = self.current_agent.train_with_get_output(state, reward=reward, done=False)

        self.previous_round_is_active = round_is_active
        # return None
        return action

    def get_state(self, rb_tick: RigidBodyTick, packet: GameTickPacket) -> np.ndarray:
        ball_state = rb_tick.ball.state
        car_state = rb_tick.players[0].state
        array = np.array([
            car_state.location.x / 4000,
            car_state.location.y / 6000,
            car_state.location.z / 400,
            car_state.velocity.x / 2000,
            car_state.velocity.y / 2000,
            car_state.velocity.z / 2000,
            car_state.rotation.x,
            car_state.rotation.y,
            car_state.rotation.z,
            car_state.rotation.w,

            ball_state.location.x / 4000,
            ball_state.location.y / 6000,
            ball_state.location.z / 400,
            ball_state.velocity.x / 2000,
            ball_state.velocity.y / 2000,
            ball_state.velocity.z / 2000,
        ])
        return array

    def get_reward(self, rb_tick: RigidBodyTick, packet: GameTickPacket, done: bool) -> float:
        if done:
            DONE_WEIGHT = 50
            ball_location = rb_tick.ball.state.location
            print(ball_location)
            # Check if scored
            if 4990 < ball_location.y < 5050 and abs(ball_location.x) < 670 and ball_location.z < 450:
                return DONE_WEIGHT
            else:
                return -DONE_WEIGHT
        reward = 0

        # Get distance from ghost
        ghost_location = self.challenger.ghost_handler.get_location(self.challenger.current_round, rb_tick)

        car_location = rb_tick.players[0].state.location
        car_location_array = np.array([car_location.x, car_location.y, car_location.z])
        distance_from_ghost = np.sqrt(((car_location_array - ghost_location) ** 2).sum())
        reward += -distance_from_ghost / 10000

        return reward

    def create_agent(self, trained_model_filepaths: Tuple[Path, Path] = None):
        INPUTS = 16
        OUTPUTS = 8  # steer, throttle, pitch, yaw, roll, jump, boost, handbrake
        if trained_model_filepaths:
            pass
        else:
            actor_model = DenseModel(inputs=INPUTS, outputs=OUTPUTS, layer_nodes=(48, 48), learning_rate=3e-3,
                                     inner_activation='relu', output_activation='tanh')
            critic_model = BaseCriticModel(inputs=INPUTS, outputs=OUTPUTS)
            self.current_agent = DDPGAgent(
                actor_model, critic_model=critic_model,
                exploration=OrnsteinUhlenbeck(theta=0.15, sigma=0.3),
            )

    @staticmethod
    def get_controller_state_from_actions(controls: np.ndarray) -> SimpleControllerState:
        controls = controls.clip(-1, 1)
        controller_state = SimpleControllerState()
        controller_state.steer = controls[0]
        controller_state.throttle = controls[1]
        controller_state.pitch = controls[2]
        controller_state.yaw = controls[3]
        controller_state.roll = controls[4]
        controller_state.jump = bool(controls[5] >= 0)
        controller_state.boost = bool(controls[6] >= 0)
        controller_state.handbrake = bool(controls[7] >= 0)
        return controller_state
