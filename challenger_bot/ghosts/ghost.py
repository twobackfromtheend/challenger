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

        self.current_round: Optional[int] = None
        self.current_ghost: Optional[pd.DataFrame] = None
        self.current_ghost_ball_start_position: Optional[np.ndarray] = None
        self.current_ghost_start_frame: Optional[int] = None

    def update(self, rbtick: RigidBodyTick):
        ball_frame = rbtick.ball.state.frame
        ball_location = rbtick.ball.state.location
        car_location = rbtick.players[0].state.location
        data_dict = {
            'ball_frame': ball_frame,
            'ball_x': ball_location.x,
            'ball_y': ball_location.y,
            'ball_z': ball_location.z,
            'x': car_location.x,
            'y': car_location.y,
            'z': car_location.z,
        }
        self.current_save.append(data_dict)

    def save_ghost(self, round: int):
        print(f"SAVING GHOST FOR ROUND: {round}")
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
        if round in self.ghosts:
            return self.ghosts[round]

        data_path: Path = self.data_folderpath / f"{round}.csv"
        if data_path.is_file():
            df = pd.read_csv(data_path, index_col=0)

            # Remove duplicate frames
            # df = df[~df.index.duplicated(keep='first')]

            # Remove repeated first frames
            df = df.drop_duplicates(keep='last')
            df.index = df.index - df.index[0]

            df = df.reindex(pd.RangeIndex(df.index.max()))
            df.interpolate('cubic', inplace=True)
            self.ghosts[round] = df
            return df
        print(f"Cannot find ghost for round: {round}")

    def get_location(self, round: int, rbtick: RigidBodyTick):
        if round is None:
            return [0, 0, 0]
        if round != self.current_round:
            self.current_ghost = self.get_ghost(round)
            self.current_ghost_ball_start_position = self.current_ghost.loc[0, ['ball_x', 'ball_y', 'ball_z']]
        ball_location = rbtick.ball.state.location
        ball_location_array = np.array([ball_location.x, ball_location.y, ball_location.z])
        ball_distance_from_start = np.sqrt(((self.current_ghost_ball_start_position - ball_location_array) ** 2).sum())

        ball_frame = rbtick.ball.state.frame
        if ball_distance_from_start < 1e-5:
            self.current_ghost_start_frame = ball_frame
        current_frame = ball_frame - self.current_ghost_start_frame
        try:
            car_position = self.current_ghost.loc[current_frame, ['x', 'y', 'z']]
        except KeyError:
            car_position = self.current_ghost.loc[self.current_ghost.index.max(), ['x', 'y', 'z']]
        return car_position.tolist()


if __name__ == '__main__':
    ghost_handler = GhostHandler()

    ghost_handler.get_location(1, None)

