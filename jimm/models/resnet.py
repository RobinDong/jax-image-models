import jax
import jax.numpy as jnp

from jax import lax


def relu(x):
    return jnp.maximum(0, x)

def max_pool(x, kernel_size: Tuple[int, int], stride: Tuple[int, int], padding="SAME"):
    return lax.reduce_window(
        x,
        -jnp.inf,
        lax.max,
        window_dimensions = (1, *kernel_size),
        window_strides = (1, *stride),
        padding = padding,
    )

class ResNet:
    def __init__(self, stages: List[int], bottleneck: bool, scale: float=1e-2):
        keys = random.split(random.PRNGKey(0), 1 + len(stages))
        self.conv1 = scale * random.normal(keys[0], (64, 3, 7, 7))
        if bottleneck:
            pass
        else:
            sub_keys = random.split(keys[1], stages[0] * 2)
            self.conv2 = [scale * random.normal(sub_key, (64, 64, 3, 3))] for sub_key in sub_keys]
            sub_keys = random.split(keys[2], stages[1] * 2)
            self.conv3 = [scale * random.normal(sub_keys[0], (128, 64, 3, 3))]
            self.conv3 += [scale * random.normal(sub_keys[idx], (128, 128, 3, 3)) for idx in range(1, len(sub_keys))]
            sub_keys = random.split(keys[3], stages[2] * 2)
            self.conv4 = [scale * random.normal(sub_keys[0], (256, 128, 3, 3))]
            self.conv4 += [scale * random.normal(sub_keys[idx], (256, 256, 3, 3)) for idx in range(1, len(sub_keys))]
            sub_keys = random.split(keys[4], stages[3] * 2)
            self.conv5 = [scale * random.normal(sub_keys[0], (512, 256, 3, 3))]
            self.conv5 += [scale * random.normal(sub_keys[idx], (512, 512, 3, 3)) for idx in range(1, len(sub_keys))]

    @staticmethod
    def infer(params, image):
        conv1, conv2, conv3, conv4, conv5 = params
        out = lax.conv_general_dilated(
            image[None, :],
            conv1,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        )
        max_pool(out, (3, 3), (2, 2))

