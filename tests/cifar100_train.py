import pickle
import functools

import jax
import jax.numpy as jnp

from jax import jit, nn, device_put
from jax.scipy.special import logsumexp
from jimm.models import ResNet


def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(ResNet.infer(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)


def loss_fn(params, images, targets):
    preds = ResNet.infer(params, images)
    preds = preds - logsumexp(preds)
    return -jnp.mean(preds * targets)


with open("cifar-100-python/train", "rb") as fp:
    batch1 = pickle.load(fp, encoding="bytes")

# Training loop
num_epochs = 1000
batch_size = 1024
learning_rate = 1e-1
dataset = device_put(jnp.array(batch1[b"data"], dtype="float32"))
dlabels = device_put(jnp.array(batch1[b"fine_labels"]))
length = len(dataset) // batch_size

loss_and_grad_fn = jit(jax.value_and_grad(loss_fn))
update_fn = jit(functools.partial(jax.tree_map, lambda p, g: p - learning_rate * g))

params = ResNet.init([2, 2, 2, 2], classes=100, bottleneck=False, scale=1e-2)  # resnet18
for name in params.keys():
    print(name, params[name].shape)

for epoch in range(num_epochs):
    train_acc = 0.0
    for start in range(length):
        begin, end = start * batch_size, (start + 1) * batch_size
        images = dataset[begin:end]
        images = jnp.reshape(images, (batch_size, 3, 32, 32))
        labels = nn.one_hot(dlabels[begin:end], 100)
        loss, grads = loss_and_grad_fn(params, images, labels)
        params = update_fn(params, grads)
        train_acc += accuracy(params, images, labels)
    print(f"[{epoch}] Train accu:", train_acc / length)
