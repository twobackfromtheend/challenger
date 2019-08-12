from typing import TYPE_CHECKING, Union

import numpy as np

from eagle_bot.reinforcement_learning.model.base_model import BaseModel

if TYPE_CHECKING:
    from tensorflow.python.keras.models import Model, Sequential
    from tensorflow.python.keras.optimizers import Optimizer


class BaseCriticModel(BaseModel):
    def __init__(self, inputs: int, outputs: int, learning_rate: float, load_from_filepath: str = None, **kwargs):
        from tensorflow.python.keras.layers import Input

        self.learning_rate = learning_rate
        self.action_input: Input = None

        super().__init__(inputs, outputs, load_from_filepath, **kwargs)

        if load_from_filepath:
            from tensorflow.python.keras.optimizers import Adam
            # Reset critic optimiser with given LR
            self.model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
            self.action_input = self.model.get_layer("action_input").input
            self.observation_input = self.model.get_layer("observation_input").input

        # After build_model is called and self.model is set.
        self.action_input_index = self.model.input.index(self.action_input)

    def build_model(self) -> Union['Sequential', 'Model']:
        raise NotImplementedError

    def _get_input_layers(self):
        from tensorflow.python.keras.layers import Flatten, Input, Concatenate

        self.action_input = Input(shape=(self.outputs,), name='action_input')
        self.observation_input = Input(shape=(self.inputs,), name='observation_input')
        inputs = [self.action_input, self.observation_input]
        x = Concatenate()(inputs)
        x = Flatten()(x)
        return inputs, x

    def create_input(self, states: np.ndarray, actions: np.ndarray):
        """
        Creates the appropriate input format for the model.
        :param states:
        :param actions:
        :return:
        """
        input_ = [states]
        input_.insert(self.action_input_index, actions)
        return input_

    def get_actor_train_fn(self, actor_model: BaseModel, actor_optimizer: 'Optimizer'):
        from tensorflow.python.keras import backend as K

        combined_inputs = []
        state_inputs = []
        for _input in self.model.input:
            if _input == self.action_input:
                combined_inputs.append([])
            else:
                combined_inputs.append(_input)
                state_inputs.append(_input)

        combined_inputs[self.action_input_index] = actor_model.model(state_inputs)

        combined_output = self.model(combined_inputs)

        updates = actor_optimizer.get_updates(
            params=actor_model.model.trainable_weights, loss=-K.mean(combined_output))

        actor_train_fn = K.function(
            state_inputs + [K.learning_phase()],
            [actor_model.model(state_inputs)],
            updates=updates
        )

        return actor_train_fn
