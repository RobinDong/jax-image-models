import pickle
import functools

import jax
import jax.numpy as jnp

from jax import jit, nn, device_put
from jax.scipy.special import logsumexp
from jimm.models import resnet34


def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(resnet34.infer(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)


def loss_fn(params, images, targets):
    preds = resnet34.infer(params, images)
    preds = preds - logsumexp(preds)
    return -jnp.mean(preds * targets)


with open("cifar-100-python/train", "rb") as fp:
    train_data = pickle.load(fp, encoding="bytes")

with open("cifar-100-python/test", "rb") as fp:
    test_data = pickle.load(fp, encoding="bytes")

# Training loop
CLASSES = 100
num_epochs = 1000
batch_size = 1024
learning_rate = 1e-1
train_ds = device_put(jnp.array(train_data[b"data"], dtype="float32"))
train_labels = device_put(jnp.array(train_data[b"fine_labels"]))
test_ds = device_put(jnp.array(test_data[b"data"], dtype="float32"))
test_images = jnp.reshape(test_ds, (len(test_ds), 3, 32, 32))
test_labels = nn.one_hot(device_put(jnp.array(test_data[b"fine_labels"])), CLASSES)

loss_and_grad_fn = jit(jax.value_and_grad(loss_fn))
update_fn = jit(functools.partial(jax.tree_map, lambda p, g: p - learning_rate * g))

params = resnet34.init(
    classes=CLASSES, scale=1e-2
)  # resnet34
for name in params.keys():
    print(name, params[name].shape)

train_len = len(train_ds) // batch_size

for epoch in range(num_epochs):
    train_acc = 0.0
    for start in range(train_len):
        begin, end = start * batch_size, (start + 1) * batch_size
        images = train_ds[begin:end]
        images = jnp.reshape(images, (batch_size, 3, 32, 32))
        labels = nn.one_hot(train_labels[begin:end], CLASSES)
        loss, grads = loss_and_grad_fn(params, images, labels)
        params = update_fn(params, grads)
        train_acc += accuracy(params, images, labels)
    test_acc = accuracy(params, test_images, test_labels)
    print(f"[{epoch}] Train accu:", train_acc / train_len, ". Test accu:", test_acc)
