from typing import TYPE_CHECKING, Sequence

from eagle_bot.reinforcement_learning.model.base_critic_model import BaseCriticModel

if TYPE_CHECKING:
    from tensorflow.python.keras import Model


class DenseCriticModel(BaseCriticModel):

    def __init__(self, inputs: int, outputs: int, learning_rate: float, load_from_filepath: str = None,
                 layer_nodes: Sequence[int] = (64, 64),
                 inner_activation='relu', output_activation='linear',
                 regularizer=None, **kwargs):
        import tensorflow as tf

        self.layer_nodes = layer_nodes
        self.inner_activation = inner_activation
        self.output_activation = output_activation
        self.regularizer = regularizer if regularizer is not None else tf.keras.regularizers.l2(1e-5)

        super().__init__(inputs, outputs, learning_rate, load_from_filepath, **kwargs)

    def build_model(self) -> 'Model':
        from tensorflow.python.keras.layers import Dense, Activation
        from tensorflow.python.keras.models import Model
        from tensorflow.python.keras.optimizers import Adam

        inputs, x = self._get_input_layers()

        for nodes in self.layer_nodes:
            x = Dense(nodes)(x)
            x = Activation(self.inner_activation)(x)

        x = Dense(self.outputs, activation=self.output_activation)(x)

        model = Model(inputs=inputs, outputs=x)

        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        # print(model.summary())
        return model
