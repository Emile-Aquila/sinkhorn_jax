import jax
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt


@jit
def entropy(P: jnp.ndarray) -> jnp.array:
    ans = P * jnp.log(P + 1e-30) - P
    return ans.sum()


@jit
def d_euclidean_matrix(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:  # (nx, d), (ny, d) -> (nx, ny)
    c = jnp.sum((x[:, None] - y[None]) ** 2, axis=-1)
    return c


@jit
def sinkhorn(s: jnp.ndarray, t: jnp.ndarray, cost_matrix: jnp.ndarray, eps: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    c = jnp.exp(-cost_matrix / eps)

    def update(_: int, val: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        v_ = t / (c.T @ val[0])
        u_ = s / (c @ v_)
        return u_, v_

    u, v = jax.lax.fori_loop(0, 500, update, (jnp.ones_like(s), jnp.ones_like(t)))
    P = u.reshape(-1, 1) * c * v.reshape(1, -1)
    div = (P * cost_matrix).sum() - eps * entropy(P)
    return div, P  # divergence, ot


@jit
def sinkhorn_log(s: jnp.ndarray, t: jnp.ndarray, cost_matrix: jnp.ndarray, eps: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    c = cost_matrix / eps / cost_matrix.max()
    ls, lt = jnp.log(s), jnp.log(t)

    def update(_: int, val: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        _g = lt - jax.scipy.special.logsumexp(val[0][:, None] - c, axis=0).squeeze()
        _f = ls - jax.scipy.special.logsumexp(_g[None, :] - c, axis=-1).squeeze()
        return _f, _g

    f, g = jax.lax.fori_loop(0, 500, update, (jnp.ones_like(s), jnp.zeros_like(t)))
    P = jnp.exp(f.reshape(-1, 1) + g.reshape(1, -1) - c)
    div = (P * cost_matrix).sum() - eps * entropy(P)
    return div, P  # divergence, ot


def plot_ot_matrix(p: jnp.ndarray) -> None:
    plt.imshow(p)
    plt.show()


def plot_point_clouds(source: jnp.ndarray, target: jnp.ndarray) -> None:
    plt.scatter(source[:, 0], source[:, 1], c='b', label="source")
    plt.scatter(target[:, 0], target[:, 1], c='r', label="target", marker="x")
    plt.legend()
    plt.show()
