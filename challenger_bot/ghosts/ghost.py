from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rlbot.utils.structures.rigid_body_struct import RigidBodyTick


class GhostHandler:
    data_folderpath = Path(__file__).parent / "data"

    def __init__(self):
        self.ghosts = {}  # round: pd.DataFrame
        self.current_save = []

        self.current_ghost: Optional[pd.DataFrame] = None
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

    def save_ghost(self, round: int):
        if not self.current_save:
            return
        print(f"Saving ghost for round: {round}")

        df = pd.DataFrame.from_records(self.current_save, index='ball_frame')
        df.to_csv(self.data_folderpath / f"{round}.csv")
        # print(df)
        # print(df.dtypes)
        # try:
        #     df.to_pickle(self.data_folderpath / f"{round}.pkl")
        # except Exception as e:
        #     print("Cannot save pickle:", e)
        self.current_save = []

    def get_ghost(self, round: int):
        self.current_ghost = self.load_saved_data(round)
        if self.current_ghost is not None:
            print(f"Found ghost for round: {round}")
            self.current_ghost_ball_start_position = self.current_ghost.loc[0, ['ball_x', 'ball_y', 'ball_z']]
            self.current_ghost_first_control_frame = self.current_ghost.loc[:, ['boost', 'throttle']].any(axis=1).idxmax()

    def load_saved_data(self, round: int):
        # if round in self.ghosts:
        #     return self.ghosts[round]

        data_path: Path = self.data_folderpath / f"{round}.csv"
        if data_path.is_file():
            df = pd.read_csv(data_path, index_col=0)

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
                self.ghosts[round] = df
                return df
        print(f"Cannot find ghost for round: {round}")

    def get_location(self, rbtick: RigidBodyTick):
        if self.current_ghost is None:
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


if __name__ == '__main__':
    ghost_handler = GhostHandler()
    ghost_handler.get_ghost(15)
    ghost_handler.get_location(1, None)

