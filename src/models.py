"""U-Net for binary building segmentation.

One architecture, deliberately. The experiment varies how imbalance is handled,
so the network has to stay fixed: if the architecture moved between arms, a
difference in recall could not be attributed to the loss function. Width and
depth are configurable only so the whole comparison can be rerun at a smaller
size on a weaker GPU.

Padding is "same" throughout and the input side must be divisible by
2 ** depth, which 256 is for depth <= 8, so skip connections line up without
cropping.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def _block(x, filters: int, dropout: float, name: str):
    for i in (1, 2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                          name=f"{name}_conv{i}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.Activation("relu", name=f"{name}_relu{i}")(x)
    if dropout:
        x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    return x


def unet(
    input_shape: tuple[int, int, int] = (256, 256, 3),
    depth: int = 4,
    base_filters: int = 32,
    dropout: float = 0.1,
    output_bias: float | None = None,
) -> keras.Model:
    """
    output_bias
        Initial bias of the final 1x1 convolution. Setting it to
        log(f / (1 - f)) for the measured positive fraction f starts the model
        at the correct base rate instead of at 0.5. Without it the first epochs
        are spent unlearning a confident wrong prior, which on a 2-5% positive
        class shows up as a loss spike and sometimes a dead run.
    """
    side = input_shape[0]
    if side % (2 ** depth):
        raise ValueError(
            f"input side {side} is not divisible by 2**depth = {2 ** depth}"
        )

    inputs = keras.Input(shape=input_shape, name="image")
    x = inputs
    skips = []
    for d in range(depth):
        f = base_filters * 2 ** d
        x = _block(x, f, dropout, f"enc{d}")
        skips.append(x)
        x = layers.MaxPooling2D(2, name=f"enc{d}_pool")(x)

    x = _block(x, base_filters * 2 ** depth, dropout, "bottleneck")

    for d in reversed(range(depth)):
        f = base_filters * 2 ** d
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same",
                                   name=f"dec{d}_up")(x)
        x = layers.Concatenate(name=f"dec{d}_cat")([x, skips[d]])
        x = _block(x, f, dropout, f"dec{d}")

    bias_init = (keras.initializers.Constant(output_bias)
                 if output_bias is not None else "zeros")
    outputs = layers.Conv2D(1, 1, activation="sigmoid",
                            bias_initializer=bias_init, name="prob")(x)
    return keras.Model(inputs, outputs, name=f"unet_d{depth}_f{base_filters}")
