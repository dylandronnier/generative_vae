from dataclasses import dataclass
from math import floor, sqrt
from typing import Optional, Tuple

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import optax
import orbax.checkpoint
from datasets import load_dataset
from flax.training.train_state import TrainState
from generative_vae.model import GLOW
from jax import Array, jit, value_and_grad, vmap
from jax.image import resize
from mpl_toolkits.axes_grid1 import ImageGrid


@vmap
def get_logpz(z, priors):
    logpz = 0
    for zi, priori in zip(z, priors):
        if priori is None:
            mu = jnp.zeros(zi.shape)
            logsigma = jnp.zeros(zi.shape)
        else:
            mu, logsigma = jnp.split(priori, 2, axis=-1)
        logpz += jnp.sum(
            -logsigma
            - 0.5 * jnp.log(2 * jnp.pi)
            - 0.5 * (zi - mu) ** 2 / jnp.exp(2 * logsigma)
        )
    return logpz


@jit
def get_logpx(z, logdets, priors):
    logpz = get_logpz(z, priors)
    logpz = jnp.mean(logpz) / bits_per_dims_norm  # bits per dimension normalization
    logdets = jnp.mean(logdets) / bits_per_dims_norm
    logpx = logpz + logdets - num_bits  # num_bits: dequantization factor
    return logpx, logpz, logdets


@jit
def postprocess(x, num_bits):
    """Map [-0.5, 0.5] quantized images to uint space"""
    num_bins = 2**num_bits
    x = jnp.floor((x + 0.5) * num_bins)
    x *= 256.0 / num_bins
    return jnp.clip(x, 0, 255).astype(jnp.uint8)


@jit
def train_step(state: TrainState, batch):
    """Training step of the model."""

    def loss_fn(params):
        _, z, logdets, priors = state.apply_fn({"params": params}, **batch)
        logpx, logpz, logdets = get_logpx(z, logdets, priors)
        return -logpx, (logpz, logdets)

    logs, grad = value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grad)
    return logs, state


@jit
def eval_step(state: TrainState, batch):
    """Evaluation step of the model."""
    _, z, logdets, priors = state.apply_fn(
        {"params": state.params}, **batch, reverse=False
    )
    return -get_logpx(z, logdets, priors)[0]


def plot_image_grid(
    y: Array,
    title: Optional[str] = None,
    display: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 9),
) -> None:
    """Plot and optionally save an image grid with matplotlib"""
    fig = plt.figure(figsize=figsize)
    num_rows = int(floor(sqrt(y.shape[0])))
    grid = ImageGrid(fig, 111, nrows_ncols=(num_rows, num_rows), axes_pad=0.1)
    for ax in grid:
        ax.set_axis_off()
    for ax, im in zip(grid, y):
        ax.imshow(im)
    if y is not None:
        fig.suptitle(title, fontsize=18)
    fig.subplots_adjust(top=0.98)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if display:
        plt.show()
    else:
        plt.close()


@jit
def map_fn(image, num_bits=5, size=256, training=True):
    """Read image file, quantize and map to [-0.5, 0.5] range.
    If num_bits = 8, there is no quantization effect.
    """
    image = resize(image, (None, size, size, None), "bilinear")
    image = jnp.clip(image, 0.0, 255.0)
    # Discretize to the given number of bits
    if num_bits < 8:
        image = jnp.floor(image / 2 ** (8 - num_bits))
    # Send to [-1, 1]
    num_bins = 2**num_bits
    image = image / num_bins - 0.5
    if training:
        image = image + random.uniform(image.shape, 0, 1.0 / num_bins)
    return image


@dataclass
class Config:
    K: int
    L: int
    init_lr: float
    num_epochs: int


if __name__ == "__main__":
    c = Config(K=16, L=3, init_lr=1e-3, num_epochs=8)

    # Load the dataset
    image_dataset = load_dataset("nielsr/CelebA-faces").shuffle()
    image_dataset = image_dataset.with_format("jax")

    plot_image_grid(next(image_dataset["train"].iter(batch_size=9))["image"])

    # Init the model
    model = GLOW()

    # Init the parameters
    rng = random.PRNGKey(0)
    rng, init_rng, default_init_rng, random_z_init_rng = random.split(rng, 4)
    exmp_imgs = next(image_dataset["train"].iter(batch_size=4))["image"]
    init_params = model.init(
        {
            "params": init_rng,
            "default": default_init_rng,
            "random_z": random_z_init_rng,
        },
        exmp_imgs,
    )["params"]

    # Init the state
    state = TrainState.create(
        apply_fn=model.apply, params=init_params, tx=optax.adam(c.init_lr)
    )

    # Checkpoints
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        "/tmp/flax_ckpt/orbax/managed", orbax_checkpointer, options
    )

    # Training loop
    for e in range(c.num_epochs):
        for batch in image_dataset["train"].iter(batch_size=32, drop_last_batch=True):
            state = train_step(state, batch["image"])

        checkpoint_manager.save(step, ckpt, save_kwargs={"save_args": save_args})
