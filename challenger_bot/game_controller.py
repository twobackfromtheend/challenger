from enum import Enum

import pyautogui
from rlbot.utils.structures.game_data_struct import GameTickPacket


def reset_shot():
    pyautogui.press('backspace')


def skip_replay():
    pyautogui.click(button='right')


class GameState(Enum):
    PAUSED = -1
    ROUND_WAITING = 1
    ROUND_ONGOING = 2
    ROUND_FINISHED = 3
    REPLAY = 4


def get_game_state(packet: GameTickPacket, previous_packet: GameTickPacket) -> GameState:
    current_seconds_elapsed = packet.game_info.seconds_elapsed
    previous_seconds_elapsed = previous_packet.game_info.seconds_elapsed
    # print(current_seconds_elapsed, previous_seconds_elapsed, previous_seconds_elapsed == current_seconds_elapsed)
    if previous_seconds_elapsed == current_seconds_elapsed:
        return GameState.PAUSED

    round_is_active = packet.game_info.is_round_active
    ball_velocity = packet.game_ball.physics.velocity
    ball_velocity_is_zero = ball_velocity.x == 0 and ball_velocity.y == 0 and ball_velocity.z == 0
    time_remaining = packet.game_info.game_time_remaining
    time_remaining_is_int = float(time_remaining).is_integer()
    previous_time_remaining = previous_packet.game_info.game_time_remaining
    # print(time_remaining, previous_time_remaining, previous_time_remaining == time_remaining)
    # print(packet.game_info)

    if round_is_active:
        return GameState.ROUND_ONGOING
    else:
        if ball_velocity_is_zero:
            if time_remaining_is_int and time_remaining != 0:
                return GameState.ROUND_WAITING
            else:
                return GameState.ROUND_FINISHED
        else:
            return GameState.REPLAY

    # if time_remaining > 0:
    #     if ball_velocity_is_zero and not round_is_active:
    #         return GameState.ROUND_WAITING
    #     else:
    #         return GameState.ROUND_ONGOING
    # else:
    #     if round_is_active:
    #         return GameState.ROUND_ONGOING
    #     else:
    #         if ball_velocity_is_zero:
    #             return GameState.ROUND_FINISHED
    #         else:
    #             return GameState.REPLAY

    # print(time_remaining)
    # print(packet.game_info)
    # if not round_is_active and (ball_velocity.x != 0 or ball_velocity.y != 0 or ball_velocity != 0):
    #     return True
    # return False
