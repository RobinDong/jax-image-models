import pickle
import functools
import numpy as np

import jax
import jax.numpy as jnp

import torch
import torchvision.transforms as T

from jax import jit, nn, device_put
from jax.scipy.special import logsumexp
from jimm.models import resnet50
from jimm.data import mixup


def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(resnet50.infer(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)


def loss_fn(params, images, targets):
    preds = resnet50.infer(params, images)
    preds = preds - logsumexp(preds)
    return -jnp.mean(preds * targets)


with open("cifar-100-python/train", "rb") as fp:
    train_data = pickle.load(fp, encoding="bytes")

with open("cifar-100-python/test", "rb") as fp:
    test_data = pickle.load(fp, encoding="bytes")

# Training loop
CLASSES = 100
num_epochs = 1000
batch_size = 8192
learning_rate = 1.0
train_ds = np.array(train_data[b"data"])
train_images = np.reshape(train_ds, (len(train_ds), 3, 32, 32))
train_labels = nn.one_hot(device_put(jnp.array(train_data[b"fine_labels"])), CLASSES)
test_ds = device_put(jnp.array(test_data[b"data"], dtype="float32"))
test_images = jnp.reshape(test_ds, (len(test_ds), 3, 32, 32))
test_labels = nn.one_hot(device_put(jnp.array(test_data[b"fine_labels"])), CLASSES)

loss_and_grad_fn = jit(jax.value_and_grad(loss_fn))
update_fn = jit(functools.partial(jax.tree_map, lambda p, g: p - learning_rate * g))

params = resnet50.init(classes=CLASSES, scale=1e-2)  # resnet50
for name in params.keys():
    print(name, params[name].shape)

train_len = len(train_ds) // batch_size
randaug = T.RandAugment().cuda()

for epoch in range(num_epochs):
    train_acc = 0.0
    key = jax.random.PRNGKey(epoch*train_len)
    keys = jax.random.split(key, train_len)
    for start in range(train_len):
        begin, end = start * batch_size, (start + 1) * batch_size
        images = train_images[begin:end]
        images = randaug(torch.tensor(images)).numpy() / 1.0
        labels = train_labels[begin:end]
        mixed_images, mixed_labels = mixup.mixup_data(images, labels, keys[start])
        loss, grads = loss_and_grad_fn(params, mixed_images, mixed_labels)
        train_acc += accuracy(params, images, labels)
        params = update_fn(params, grads)
    test_acc = accuracy(params, test_images/1.0, test_labels)
    print(f"[{epoch}] Train accu:", train_acc/train_len, ". Test accu:", test_acc)
