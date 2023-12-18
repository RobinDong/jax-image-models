import jax.numpy as jnp
import jax.nn as nn

from jax import lax, random


def max_pool(x, kernel_size: tuple[int, int], stride: tuple[int, int], padding="SAME"):
    return lax.reduce_window(
        x,
        -jnp.inf,
        lax.max,
        window_dimensions=(1, 1, *kernel_size),
        window_strides=(1, 1, *stride),
        padding=padding,
    )


def avg_pool(x, kernel_size: tuple[int, int], stride: tuple[int, int], padding="SAME"):
    out = lax.reduce_window(
        x,
        0.0,
        lax.add,
        window_dimensions=(1, 1, *kernel_size),
        window_strides=(1, 1, *stride),
        padding=padding,
    )
    ones = jnp.ones(x.shape)
    ones_out = lax.reduce_window(
        ones,
        0.0,
        lax.add,
        window_dimensions=(1, 1, *kernel_size),
        window_strides=(1, 1, *stride),
        padding=padding,
    )
    return out / ones_out


def batch_norm(params, inputs):
    gamma, beta = params
    mean = jnp.mean(inputs, axis=(0, 2, 3), keepdims=True)
    var = jnp.mean((inputs - mean) ** 2, axis=(0, 2, 3), keepdims=True)
    var += 1e-5
    stddev = jnp.sqrt(var)
    out = (inputs - mean) / stddev
    return gamma * out + beta


class ResNet:
    @staticmethod
    def init(stages: list[int], bottleneck: bool, scale: float = 1e-2):
        keys = random.split(random.PRNGKey(0), 1 + len(stages) + 1)
        sub_keys = random.split(keys[0], 2)
        conv1 = scale * random.normal(sub_keys[0], (64, 3, 7, 7))
        bn1 = scale * random.normal(sub_keys[1], (2,))
        params = {"conv1": conv1, "bn1": bn1}
        if bottleneck:
            pass
        else:
            sub_keys = random.split(keys[1], stages[0] * 8)
            # conv2
            layers = stages[0] * 4
            for index in range(0, layers, 2):
                conv = scale * random.normal(sub_keys[index], (64, 64, 3, 3))
                bn = scale * random.normal(sub_keys[index + 1], (2,))
                params |= {f"conv2_{index//2+1}": conv, f"bn2_{index//2+1}": bn}
            # conv3 conv4 conv5
            for index, depth in [(1, 128), (2, 256), (3, 512)]:
                sub_keys = random.split(keys[index + 1], stages[index] * 10)
                layers = stages[index] * 4
                for sub_index in range(0, layers, 2):
                    if sub_index == 0:  # first conv layer of this stage
                        conv = scale * random.normal(
                            sub_keys[sub_index], (depth, depth // 2, 3, 3)
                        )
                    else:
                        conv = scale * random.normal(
                            sub_keys[sub_index], (depth, depth, 3, 3)
                        )
                    bn = scale * random.normal(sub_keys[sub_index + 1], (2,))
                    params |= {
                        f"conv{index+2}_{sub_index//2+1}": conv,
                        f"bn{index+2}_{sub_index//2+1}": bn,
                    }
                downsample = scale * random.normal(
                    sub_keys[8], (depth, depth // 2, 1, 1)
                )
                bn = scale * random.normal(sub_keys[9], (2,))
                params |= {f"downsample{index+2}": downsample, f"bn{index+2}": bn}
        sub_keys = random.split(keys[-1], 2)
        n, m = 1000, 512  # classification for imagenet
        mlp_w = scale * random.normal(sub_keys[0], (n, m))
        mlp_b = scale * random.normal(sub_keys[1], (n,))
        params |= {"mlp_w": mlp_w, "mlp_b": mlp_b}
        return params

    @staticmethod
    def basic_blocks(params, inp):
        out = inp
        for stage in range(2, 6):
            layers = len([name for name in params.keys() if name[:6] == f"conv{stage}_"]) // 2
            print("stage:", stage, layers)
            for layer in (1, layers+1):
                shortcut = out
                out = lax.conv_general_dilated(
                    out,
                    params[f"conv{stage}_{layer}"],
                    window_strides=(2, 2) if stage > 2 and layer == 1 else (1, 1),
                    padding="SAME",
                    dimension_numbers=("NCHW", "OIHW", "NCHW"),
                )
                out = batch_norm(params[f"bn{stage}_{layer}"], out)
                out = nn.relu(out)
                out = lax.conv_general_dilated(
                    out,
                    params[f"conv{stage}_{layer+1}"],
                    window_strides=(1, 1),
                    padding="SAME",
                    dimension_numbers=("NCHW", "OIHW", "NCHW"),
                )
                out = batch_norm(params[f"bn{stage}_{layer+1}"], out)
                if stage != 2 and layer == 1:
                    shortcut = lax.conv_general_dilated(
                        shortcut,
                        params[f"downsample{stage}"],
                        window_strides=(2, 2),
                        padding="SAME",
                        dimension_numbers=("NCHW", "OIHW", "NCHW"),
                    )
                    shortcut = batch_norm(params[f"bn{stage}"], shortcut)
                out += shortcut
                out = nn.relu(out)
                print(f"out{stage}_{layer}:", out.shape)
        return out

    @staticmethod
    def infer(params, images):
        print("params:", params.keys())
        for name, value in params.items():
            print(name, value.shape)
        # stem
        out = lax.conv_general_dilated(
            images,
            params["conv1"],
            window_strides=(2, 2),
            padding="SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        out = batch_norm(params["bn1"], out)
        out = nn.relu(out)
        out = max_pool(out, (3, 3), (2, 2))
        out = ResNet.basic_blocks(params, out)
        out = avg_pool(out, (7, 7), (7, 7))
        out = lax.reshape(out, (out.shape[0], 512))
        out = lax.dot_general(out, params["mlp_w"], ((1, 1), ((), ())),)
        return nn.softmax(out)
