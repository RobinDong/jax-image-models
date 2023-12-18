import jax
import jax.numpy as jnp
import jax.nn as nn

from jax import lax, random

def max_pool(x, kernel_size: tuple[int, int], stride: tuple[int, int], padding="SAME"):
    return lax.reduce_window(
        x,
        -jnp.inf,
        lax.max,
        window_dimensions = (1, 1, *kernel_size),
        window_strides = (1, 1, *stride),
        padding = padding,
    )

def batch_norm(params, inputs):
    gamma, beta = params
    mean = output = jnp.mean(inputs, axis=(0, 2, 3), keepdims=True)
    var = jnp.mean((inputs - mean) ** 2, axis=(0, 2, 3), keepdims=True)
    var += 1e-5
    stddev = jnp.sqrt(var)
    out = (inputs - mean) / stddev
    return gamma * out + beta


class ResNet:
    @staticmethod
    def init(stages: list[int], bottleneck: bool, scale: float=1e-2):
        keys = random.split(random.PRNGKey(0), 1 + len(stages))
        sub_keys = random.split(keys[0], 2)
        conv1 = scale * random.normal(sub_keys[0], (64, 3, 7, 7))
        bn1 = scale * random.normal(sub_keys[1], (2,))
        params = [conv1, bn1]
        if bottleneck:
            pass
        else:
            sub_keys = random.split(keys[1], stages[0] * 8)
            # conv2
            for index in range(0, 8, 2):
                if index == 6:  # last conv layer of this stage
                    conv = scale * random.normal(sub_keys[index], (64, 128, 3, 3))
                else:
                    conv = scale * random.normal(sub_keys[index], (64, 64, 3, 3))
                bn = scale * random.normal(sub_keys[index+1], (2,))
                params += [conv, bn]
            # conv3 conv4 conv5
            for index, depth in [(1, 128), (2, 256), (3, 512)]:
                sub_keys = random.split(keys[index+1], stages[index] * 8)
                for sub_index in range(0, 8, 2):
                    if sub_index == 6 and depth != 512: # last conv layer of this stage
                        conv = scale * random.normal(sub_keys[sub_index], (depth, depth * 2, 3, 3))
                    else:
                        conv = scale * random.normal(sub_keys[sub_index], (depth, depth, 3, 3))
                    bn = scale * random.normal(sub_keys[sub_index+1], (2,))
                    params += [conv, bn]
        return params

    @staticmethod
    def basic_blocks(params, inp):
        out = shortcut = inp
        for layer in range(2, len(params), 4):
            print("layer ----")
            print(params[layer].shape)
            print(params[layer+1].shape)
            print(params[layer+2].shape)
            print(params[layer+3].shape)
            out = lax.conv_general_dilated(
                out,
                params[layer],
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
            )
            out = batch_norm(params[layer+1], out)
            out = nn.relu(out)
            out = lax.conv_general_dilated(
                out,
                params[layer+2],
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
            )
            out = batch_norm(params[layer+3], out)
            out += shortcut
            out = nn.relu(out)
        return out

    @staticmethod
    def infer(params, images):
        conv1, bn1 = params[0], params[1]
        # stem
        out = lax.conv_general_dilated(
            images,
            conv1,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        )
        out = batch_norm(bn1, out)
        out = nn.relu(out)
        print("out:", out.shape)
        out = max_pool(out, (3, 3), (2, 2))
        return ResNet.basic_blocks(params, out)
