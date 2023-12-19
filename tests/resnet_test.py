import jax
import jax.numpy as jnp

import jimm
from jimm.models import ResNet

def create_resnet18():
    params = ResNet.init([2, 2, 2, 2], bottleneck=False)  # resnet18
    samples = jnp.ones((32, 3, 224, 224))
    output = ResNet.infer(params, samples)
    print("resnet18 output:", output.shape)

def create_resnet34():
    params = ResNet.init([3, 4, 6, 3], bottleneck=False)  # resnet34
    samples = jnp.ones((32, 3, 224, 224))
    output = ResNet.infer(params, samples)
    print("resnet34 output:", output.shape)

create_resnet34()
