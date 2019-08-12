from typing import Sequence, TYPE_CHECKING

from eagle_bot.reinforcement_learning.model.base_model import BaseModel

if TYPE_CHECKING:
    from tensorflow.python.keras import Sequential


class DenseActorModel(BaseModel):

    def __init__(self, inputs: int, outputs: int, load_from_filepath: str = None,
                 layer_nodes: Sequence[int] = (64, 64),
                 inner_activation='relu', output_activation='linear',
                 regularizer=None,
                 **kwargs):
        import tensorflow as tf

        self.layer_nodes = layer_nodes
        self.inner_activation = inner_activation
        self.output_activation = output_activation
        self.regularizer = regularizer if regularizer is not None else tf.keras.regularizers.l2(1e-5)

        super().__init__(inputs, outputs, load_from_filepath=load_from_filepath)

    def build_model(self) -> 'Sequential':
        from tensorflow import keras

        model = keras.Sequential()

        # Verbose version needed because https://github.com/tensorflow/tensorflow/issues/22837#issuecomment-428327601
        # model.add(keras.layers.Dense(self.inputs))
        model.add(keras.layers.Dense(input_shape=(self.inputs,), units=self.inputs))

        for _layer_nodes in self.layer_nodes:
            model.add(
                keras.layers.Dense(_layer_nodes, activation=self.inner_activation, kernel_regularizer=self.regularizer)
            )

        model.add(keras.layers.Dense(self.outputs, activation=self.output_activation))
        return model
