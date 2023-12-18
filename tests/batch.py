import pickle
import functools
import jax
import jax.numpy as jnp
import jax.nn as nn
from functools import partial
from jax import grad, jit, lax
from jax import random, device_put
from jax.scipy.special import logsumexp

class CNN:
    def __init__(self, images):
        keys = random.split(random.PRNGKey(0), 3)
        scale = 1e-2
        self.conv1 = scale * random.normal(keys[0], (3, 3, 4, 4))
        self.conv2 = scale * random.normal(keys[1], (16, 3, 4, 4))

        w_key, b_key = random.split(keys[2])
        n, m = 10, 10816
        self.mlp_w = scale * random.normal(w_key, (n, m))
        self.mlp_b = scale * random.normal(b_key, (n,))
        self.params = {"conv1": self.conv1, "conv2": self.conv2, "mlp_w": self.mlp_w, "mlp_b": self.mlp_b}

    @staticmethod
    def infer(params, images):
        activations = lax.conv_general_dilated(
            images,
            params["conv1"],
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        )
        activations = jnp.maximum(0, activations)
        #print("activations:", activations.shape)
        activations = lax.conv_general_dilated(
            activations,
            params["conv2"],
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        )
        activations = jnp.maximum(0, activations)
        activations = activations.reshape((images.shape[0], -1))
        #print("activations:", activations.shape)
        idim = activations.shape[0] * activations.shape[1]
        outputs = lax.dot_general(
            activations,
            params["mlp_w"],
            ((1, 1), ((), ())),
        ) + params["mlp_b"]
        logits = jnp.maximum(0, outputs)
        return logits - logsumexp(logits)


learning_rate = 0.001
batch_size = 256
samples = jnp.ones((batch_size, 3, 32, 32))
net = CNN(samples)

# This works on single examples
with open("/home/robin/Downloads/cifar-10-batches-py/data_batch_1", "rb") as fp:
    batch1 = pickle.load(fp, encoding='bytes')
    x = jnp.array(batch1[b"labels"])

def accuracy(params, images, targets):
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(CNN.infer(params, images), axis=1)
  return jnp.mean(predicted_class == target_class)

def loss_fn(params, images, targets):
    preds = CNN.infer(params, images)
    return -jnp.mean(preds * targets)

loss_and_grad_fn = jit(jax.value_and_grad(loss_fn))

update_fn = jit(
    functools.partial(jax.tree_map, lambda p, g: p - learning_rate * g)
)

# Training loop
num_epochs = 1000
dataset = device_put(jnp.array(batch1[b"data"], dtype="float32"))
dlabels = device_put(jnp.array(batch1[b"labels"]))
length = len(dataset) // batch_size
for epoch in range(num_epochs):
    train_acc = 0.0
    for start in range(length):
        begin, end = start * batch_size, (start + 1) * batch_size
        images = dataset[begin: end]
        images = jnp.reshape(images, (batch_size, 3, 32, 32))
        labels = nn.one_hot(dlabels[begin: end], 10)
        loss, grads = loss_and_grad_fn(net.params, images, labels)
        net.params = update_fn(net.params, grads)
        train_acc += accuracy(net.params, images, labels)
    print(f"[{epoch}] Train accu:", train_acc / length)
