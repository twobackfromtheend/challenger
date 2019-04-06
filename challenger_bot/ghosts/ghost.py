import random
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.structures.rigid_body_struct import RigidBodyTick


class GhostHandler:
    data_folderpath = Path(__file__).parent / "data"

    def __init__(self, challenge: str):
        self.challenge = challenge
        self.challenge_data_folderpath: Path = self.data_folderpath / challenge
        self.challenge_data_folderpath.mkdir(exist_ok=True)

        self.current_save = []

        self.current_ghost: Optional[pd.DataFrame] = None
        self.current_ghosts: Optional[List[pd.DataFrame]] = []
        self.current_ghost_ball_start_position: Optional[np.ndarray] = None
        self.current_ghost_first_control_frame: Optional[int] = None
        self.replay_ghost_start_frame: Optional[int] = None

    def update(self, rbtick: RigidBodyTick):
        ball_frame = rbtick.ball.state.frame
        ball_location = rbtick.ball.state.location
        car_location = rbtick.players[0].state.location
        car_inputs = rbtick.players[0].input
        data_dict = {
            'ball_frame': ball_frame,
            'ball_x': ball_location.x,
            'ball_y': ball_location.y,
            'ball_z': ball_location.z,
            'x': car_location.x,
            'y': car_location.y,
            'z': car_location.z,
            'throttle': car_inputs.throttle,
            'steer': car_inputs.steer,
            'pitch': car_inputs.pitch,
            'yaw': car_inputs.yaw,
            'roll': car_inputs.roll,
            'jump': car_inputs.jump,
            'boost': car_inputs.boost,
            'handbrake': car_inputs.handbrake,
        }
        self.current_save.append(data_dict)

    def save_ghost(self, round_: int):
        if not self.current_save:
            return
        save_location = self.get_save_location(round_)
        print(f"Saving ghost for round: {round_} ({save_location.relative_to(Path(__file__).parent)})")
        df = pd.DataFrame.from_records(self.current_save, index='ball_frame')
        df.to_csv(save_location)
        self.current_save = []

    def get_save_location(self, round_: int):
        round_folder = self.get_round_folder(round_)
        round_folder.mkdir(exist_ok=True)
        max_i = 0
        for file in round_folder.glob("*.csv"):
            filename = file.name
            i = int(filename[:filename.index('.')])
            max_i = max(i + 1, max_i)

        return round_folder / f"{max_i}.csv"

    def get_round_folder(self, round_) -> Path:
        round_folder: Path = self.challenge_data_folderpath / str(round_)
        return round_folder

    def get_ghosts(self, round_: int):
        self.current_ghosts = self.load_saved_ghosts(round_)
        if self.current_ghosts:
            print(f"Found ghost for round: {round_}")
            self.randomise_current_ghost()
            self.current_ghost_ball_start_position = self.current_ghost.loc[0, ['ball_x', 'ball_y', 'ball_z']]
            self.current_ghost_first_control_frame = self.current_ghost.loc[:, ['boost', 'throttle']].any(
                axis=1).idxmax()

    def load_saved_ghosts(self, round_: int):
        round_folder = self.get_round_folder(round_)
        ghosts = []
        for file in round_folder.glob("*.csv"):
            df = pd.read_csv(file, index_col=0)

            # Remove duplicate frames
            # df = df[~df.index.duplicated(keep='first')]

            # Remove repeated first frames
            df = df.drop_duplicates(keep='last')
            df = df[~df.index.duplicated()]
            df.index = df.index - df.index[0]

            df = df.reindex(pd.RangeIndex(df.index.max()))
            df.interpolate('linear', inplace=True)
            df.fillna(method='ffill', inplace=True)  # Interpolate boolean
            if not df.empty:
                ghosts.append(df)
        if not ghosts:
            print(f"Cannot find ghosts for round: {round_}")
        return ghosts

    def randomise_current_ghost(self):
        self.current_ghost = random.choice(self.current_ghosts)

    def get_location(self, rbtick: RigidBodyTick):
        if not self.current_ghosts:
            return [0, 0, 0]

        current_frame_delta = self.get_current_frame_delta(rbtick)
        car_position = self.current_ghost.loc[current_frame_delta, ['x', 'y', 'z']]
        return car_position.tolist()

    def get_current_frame_delta(self, rb_tick: RigidBodyTick) -> int:
        car_velocity = rb_tick.players[0].state.velocity
        car_velocity_is_zero = abs(car_velocity.x) < 1 and abs(car_velocity.y) < 1 and abs(car_velocity.z) < 10
        ball_location = rb_tick.ball.state.location
        ball_location_array = np.array([ball_location.x, ball_location.y, ball_location.z])
        ball_distance_from_start = np.sqrt(((self.current_ghost_ball_start_position - ball_location_array) ** 2).sum())

        # print(ball_location_array, self.current_ghost_ball_start_position)
        # print(ball_distance_from_start)
        ball_frame = rb_tick.ball.state.frame
        if ball_distance_from_start < 1e-5 and car_velocity_is_zero:
            self.replay_ghost_start_frame = ball_frame
        if self.replay_ghost_start_frame is None:
            print("Have not found replay ghost start frame.")
            return ball_frame
        current_frame_delta = ball_frame - self.replay_ghost_start_frame
        return min(current_frame_delta + self.current_ghost_first_control_frame, self.current_ghost.index.max())

    def get_ghost_controller_state(self, rb_tick: RigidBodyTick):
        current_frame_delta = self.get_current_frame_delta(rb_tick)
        ghost = self.current_ghost
        controls = ghost.loc[
            current_frame_delta,
            ['steer', 'throttle', 'pitch', 'yaw', 'roll', 'boost', 'jump', 'handbrake']
        ]
        controls = controls.clip(-1, 1)
        controller_state = SimpleControllerState()
        controller_state.steer, controller_state.throttle, \
            controller_state.pitch, controller_state.yaw, controller_state.roll, \
            controller_state.boost, controller_state.jump, controller_state.handbrake = controls
        return controller_state


if __name__ == '__main__':
    ghost_handler = GhostHandler("7657-2F43-9B3A-C1F1")
    ghost_handler.get_ghosts(15)
    ghost_handler.get_location(1)
