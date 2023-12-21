import jax
import jax.numpy as jnp

from jax import random


@jax.jit
def mixup_data(x, y, key, alpha=1.0):
    if alpha > 0:
        lam = random.beta(key, alpha, alpha)
    else:
        lam = 1

    mixed_x = lam * x + (1 - lam) * random.permutation(key, x)
    mixed_y = lam * y + (1 - lam) * random.permutation(key, y)
    return mixed_x, mixed_y
