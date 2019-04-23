import time
from collections import defaultdict
from enum import Enum
from typing import Tuple, Dict, Sequence, Union

import contextlib
with contextlib.redirect_stdout(None):
    import pygame

from eagle_bot.ds4_interfacer.callbacks import Callback

pygame.init()
pygame.joystick.init()


class DS4Button(Enum):
    SQUARE = 0
    X = 1
    O = 2
    TRIANGLE = 3
    L1 = 4
    R1 = 5
    L2 = 6
    R2 = 7
    SHARE = 8
    OPTION = 9
    L_JOYSTICK = 10
    R_JOYSTICK = 11
    PLAYSTATION = 12
    TOUCHPAD = 13


class DS4Analog(Enum):
    # -1 (top left) to 1 (bottom right)
    L_HORIZONTAL = 0
    L_VERTICAL = 1
    R_HORIZONTAL = 2
    R_VERTICAL = 3
    # -1 (unpressed) to 1 (fully pressed)
    R2 = 4
    L2 = 5


class DS4:
    def __init__(self, callbacks: Sequence['Callback'] = ()):
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()

        self.axis_data: Dict[int, float] = {}
        self.button_data: Dict[int, bool] = defaultdict(lambda: False)
        self.dpad: Tuple[int, int] = (0, 0)

        self.last_updated_time = time.time()
        self.callbacks = callbacks

    def tick(self):
        time_now = time.time()
        if time_now - self.last_updated_time > 0.001:
            self.update()
            for callback in self.callbacks:
                callback.call(button_pressed=self.button_data[callback.button.value])
            self.last_updated_time = time_now

    def update(self):
        # print(pygame.event.get())
        for event in pygame.event.get():
            # print(event)
            if event.type == pygame.JOYAXISMOTION:
                self.axis_data[event.axis] = event.value
            elif event.type == pygame.JOYBUTTONDOWN:
                self.button_data[event.button] = True
            elif event.type == pygame.JOYBUTTONUP:
                self.button_data[event.button] = False
            elif event.type == pygame.JOYHATMOTION:
                self.dpad = event.value
        # print(self.button_data)

    def get_button(self, ds4_control: Union[DS4Button, DS4Analog]):
        if isinstance(ds4_control, DS4Button):
            return self.button_data[ds4_control.value]
        else:
            return self.axis_data.get(ds4_control.value, self.get_axis_defaults(ds4_control))

    @staticmethod
    def get_axis_defaults(control: DS4Analog):
        if control == DS4Analog.R2 or control == DS4Analog.L2:
            return -1
        else:
            return 0

if __name__ == '__main__':
    def print_o_callback():
        print('O')


    def print_x_callback():
        print('x')


    callbacks = [Callback(DS4Button.O, print_o_callback), Callback(DS4Button.X, print_x_callback, False)]
    ds4 = DS4(callbacks)

    while True:
        ds4.tick()
