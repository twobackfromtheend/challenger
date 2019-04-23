from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from eagle_bot.ds4_interfacer.ds4_controller import DS4Button


class Callback:
    def __init__(self, button: 'DS4Button', callback: Callable, fire_once: bool = True):
        self.button = button
        self.callback = callback
        self.fire_once = fire_once

        self.just_fired = False

    def call(self, button_pressed: bool):
        if button_pressed:
            if self.fire_once and self.just_fired:
                return
            self.callback()
            self.just_fired = True
        else:
            self.just_fired = False
