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
    def _create_basic_block_params(keys, stages, scale):
        params = {}
        # conv2
        layers = stages[0] * 2
        sub_keys = random.split(keys[1], layers)
        for index in range(layers):
            conv = scale * random.normal(sub_keys[index], (64, 64, 3, 3))
            bn = jnp.array([1.0, 0.0])
            params |= {f"conv2_{index+1}": conv, f"bn2_{index+1}": bn}
        # conv3 conv4 conv5
        for index, depth in [(1, 128), (2, 256), (3, 512)]:
            layers = stages[index] * 2
            sub_keys = random.split(keys[index + 1], layers + 1)
            for sub_index in range(layers):
                if sub_index == 0:  # first conv layer of this stage
                    conv = scale * random.normal(
                        sub_keys[sub_index], (depth, depth // 2, 3, 3)
                    )
                else:
                    conv = scale * random.normal(
                        sub_keys[sub_index], (depth, depth, 3, 3)
                    )
                bn = jnp.array([1.0, 0.0])
                params |= {
                    f"conv{index+2}_{sub_index+1}": conv,
                    f"bn{index+2}_{sub_index+1}": bn,
                }
            downsample = scale * random.normal(sub_keys[-1], (depth, depth // 2, 1, 1))
            bn = jnp.array([1.0, 0.0])
            params |= {f"downsample{index+2}": downsample, f"bn{index+2}": bn}
        return params

    @staticmethod
    def _create_bottle_neck_params(keys, stages, scale):
        params = {}
        # conv2
        layers = stages[0] * 3
        sub_keys = random.split(keys[1], layers + 1)
        for index in range(0, layers, 3):
            if index == 0:
                conv = scale * random.normal(sub_keys[index], (64, 64, 1, 1))
            else:
                conv = scale * random.normal(sub_keys[index], (64, 4 * 64, 1, 1))
            bn = jnp.array([1.0, 0.0])
            params |= {f"conv2_{index+1}": conv, f"bn2_{index+1}": bn}
            conv = scale * random.normal(sub_keys[index + 1], (64, 64, 3, 3))
            bn = jnp.array([1.0, 0.0])
            params |= {f"conv2_{index+2}": conv, f"bn2_{index+2}": bn}
            conv = scale * random.normal(sub_keys[index + 2], (256, 64, 1, 1))
            bn = jnp.array([1.0, 0.0])
            params |= {f"conv2_{index+3}": conv, f"bn2_{index+3}": bn}
        downsample = scale * random.normal(sub_keys[-1], (256, 64, 1, 1))
        bn = jnp.array([1.0, 0.0])
        params |= {"downsample2": downsample, "bn2": bn}
        # conv3 conv4 conv5
        for index, depth in [(1, 128), (2, 256), (3, 512)]:
            layers = stages[index] * 3
            sub_keys = random.split(keys[index + 1], layers + 1)
            for sub_index in range(0, layers, 3):
                if sub_index == 0:  # first conv layer of this stage
                    conv = scale * random.normal(
                        sub_keys[sub_index], (depth, depth * 2, 1, 1)
                    )
                elif sub_index >= 3 and (sub_index % 3) == 0:
                    conv = scale * random.normal(
                        sub_keys[sub_index], (depth, depth * 4, 1, 1)
                    )
                else:
                    conv = scale * random.normal(
                        sub_keys[sub_index], (depth, depth, 1, 1)
                    )
                bn = jnp.array([1.0, 0.0])
                params |= {
                    f"conv{index+2}_{sub_index+1}": conv,
                    f"bn{index+2}_{sub_index+1}": bn,
                }
                conv = scale * random.normal(
                    sub_keys[sub_index + 1], (depth, depth, 3, 3)
                )
                bn = jnp.array([1.0, 0.0])
                params |= {
                    f"conv{index+2}_{sub_index+2}": conv,
                    f"bn{index+2}_{sub_index+2}": bn,
                }
                conv = scale * random.normal(
                    sub_keys[sub_index + 2], (4 * depth, depth, 1, 1)
                )
                bn = jnp.array([1.0, 0.0])
                params |= {
                    f"conv{index+2}_{sub_index+3}": conv,
                    f"bn{index+2}_{sub_index+3}": bn,
                }
            downsample = scale * random.normal(
                sub_keys[-1], (depth * 4, depth * 2, 1, 1)
            )
            bn = jnp.array([1.0, 0.0])
            params |= {f"downsample{index+2}": downsample, f"bn{index+2}": bn}
        return params

    @staticmethod
    def init(
        stages: list[int], bottleneck: bool, classes: int = 1000, scale: float = 1e-2
    ):
        keys = random.split(random.PRNGKey(0), 1 + len(stages) + 1)
        conv1 = scale * random.normal(keys[0], (64, 3, 7, 7))
        bn1 = jnp.array([1.0, 0.0])
        params = {"conv1": conv1, "bn1": bn1}
        if bottleneck:
            params |= ResNet._create_bottle_neck_params(keys, stages, scale)
        else:
            params |= ResNet._create_basic_block_params(keys, stages, scale)
        sub_keys = random.split(keys[-1], 2)
        # classification for imagenet
        if bottleneck:
            filters = 2048
        else:
            filters = 512
        mlp_w = scale * random.normal(sub_keys[0], (classes, filters))
        mlp_b = scale * random.normal(sub_keys[1], (classes,))
        params |= {"mlp_w": mlp_w, "mlp_b": mlp_b}
        return params

    @staticmethod
    def basic_blocks(params, inp):
        out = inp
        for stage in range(2, 6):
            layers = (
                len([name for name in params.keys() if name[:6] == f"conv{stage}_"])
                // 2
            )
            for layer in range(1, layers + 1, 2):
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
        return out

    @staticmethod
    def bottle_neck(params, inp):
        out = inp
        for stage in range(2, 6):
            layers = (
                len([name for name in params.keys() if name[:6] == f"conv{stage}_"])
                // 3
            )
            for layer in range(1, layers + 1, 3):
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
                out = nn.relu(out)
                out = lax.conv_general_dilated(
                    out,
                    params[f"conv{stage}_{layer+2}"],
                    window_strides=(1, 1),
                    padding="SAME",
                    dimension_numbers=("NCHW", "OIHW", "NCHW"),
                )
                out = batch_norm(params[f"bn{stage}_{layer+2}"], out)
                if layer == 1:
                    shortcut = lax.conv_general_dilated(
                        shortcut,
                        params[f"downsample{stage}"],
                        window_strides=(2, 2) if stage > 2 else (1, 1),
                        padding="SAME",
                        dimension_numbers=("NCHW", "OIHW", "NCHW"),
                    )
                    shortcut = batch_norm(params[f"bn{stage}"], shortcut)
                out += shortcut
                out = nn.relu(out)
        return out

    @staticmethod
    def infer(params, images):
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
        if params["conv5_1"].shape[-1] == 3:  # 3x3 filter means bottleneck is False
            out = ResNet.basic_blocks(params, out)
        else:  # 1x1 filters means bottleneck is False
            out = ResNet.bottle_neck(params, out)
        out = avg_pool(out, (7, 7), (7, 7))
        out = out.reshape((images.shape[0], -1))
        out = lax.dot_general(
            out,
            params["mlp_w"],
            ((1, 1), ((), ())),
        )
        return out


class resnet18:
    @staticmethod
    def init(classes: int = 1000, scale: float = 1e-2):
        return ResNet.init([2, 2, 2, 2], bottleneck=False, classes=classes, scale=scale)

    @staticmethod
    def infer(params, images):
        return ResNet.infer(params, images)


class resnet34:
    @staticmethod
    def init(classes: int = 1000, scale: float = 1e-2):
        return ResNet.init([3, 4, 6, 3], bottleneck=False, classes=classes, scale=scale)

    @staticmethod
    def infer(params, images):
        return ResNet.infer(params, images)


class resnet50:
    @staticmethod
    def init(classes: int = 1000, scale: float = 1e-2):
        return ResNet.init([3, 4, 6, 3], bottleneck=True, classes=classes, scale=scale)

    @staticmethod
    def infer(params, images):
        return ResNet.infer(params, images)


class resnet101:
    @staticmethod
    def init(classes: int = 1000, scale: float = 1e-2):
        return ResNet.init([3, 4, 23, 3], bottleneck=True, classes=classes, scale=scale)

    @staticmethod
    def infer(params, images):
        return ResNet.infer(params, images)
