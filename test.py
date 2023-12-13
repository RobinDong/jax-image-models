import time
import pickle
import functools
import jax
import jax.numpy as jnp
import jax.nn as nn
from jax import grad, jit, vmap, lax
from jax import random, device_put

def save_ckpt(params):
    with open("hello.pickle", "wb") as fp:
        pickle.dump(params, fp)

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(key):
  keys = random.split(key, 3)
  conv_params1 = 1e-2 * random.normal(keys[0], (3, 1, 4, 4))
  conv_params2 = 1e-2 * random.normal(keys[1], (16, 3, 4, 4))
  mlp_params = random_layer_params(576, 10, keys[2])
  return [conv_params1, conv_params2, mlp_params]

image_size = 3 * 32 * 32
layer_sizes = [image_size, 512, 256, 10]
learning_rate = 0.001
batch_size = 256
n_targets = 10
params = init_network_params(random.PRNGKey(0))
params = device_put(params)

from jax.scipy.special import logsumexp

def relu(x):
  return jnp.maximum(0, x)

def predict(params, image):
  # per-example predictions
  conv_params1, conv_params2, mlp_params = params
  #print("image:", image.shape)

  activations = lax.conv_general_dilated(
    image[None, :],
    conv_params1,
    window_strides=(2, 2),
    padding="VALID",
    dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
    feature_group_count=3
  )
  activations = relu(activations)
  #print("activations:", activations.shape)
  activations = lax.conv_general_dilated(
    activations,
    conv_params2,
    window_strides=(2, 2),
    padding="VALID",
    dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
  )
  activations = relu(activations)
  w, b = mlp_params
  outputs = jnp.dot(w, jnp.ravel(activations)) + b
  logits = relu(outputs)
  return logits - logsumexp(logits)
  #return nn.softmax(logits)

# This works on single examples
with open("/home/robin/Downloads/cifar-10-batches-py/data_batch_1", "rb") as fp:
    batch1 = pickle.load(fp, encoding='bytes')
    x = jnp.array(batch1[b"labels"])

random_flattened_images = random.normal(random.PRNGKey(1), (batch_size, image_size))
batched_predict = vmap(predict, in_axes=(None, 0))

def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)
  
def accuracy(params, images, targets):
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
  return jnp.mean(predicted_class == target_class)

def loss_fn(params, images, targets):
  preds = batched_predict(params, images)
  return -jnp.mean(preds * targets)

loss_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

update_fn = jax.jit(
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
        labels = one_hot(dlabels[begin: end], 10)
        loss, grads = loss_and_grad_fn(params, images, labels)
        params = update_fn(params, grads)
        train_acc += accuracy(params, images, labels)
    print(f"[{epoch}] Train accu:", train_acc / length)
