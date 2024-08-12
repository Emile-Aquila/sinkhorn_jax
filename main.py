import jax
import jax.numpy as jnp
from jax import jit
from matplotlib.animation import FuncAnimation

from sinkhorn import d_euclidean_matrix, sinkhorn_log, plot_point_clouds
import matplotlib.pyplot as plt


if __name__ == '__main__':
    ns, nt, dim = 10, 10, 2
    epsilon = 0.001
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2 = jax.random.split(key, 3)

    # Generate random point clouds
    source_points = jax.random.normal(subkey1, (ns, dim))
    source_points_start = source_points
    target_points = jax.random.normal(subkey2, (nt, dim)) + jnp.array([1, 1]) * 5
    print("start calculation")

    # gradient function
    f_pc2wd = lambda x, y: (
        sinkhorn_log(
            jnp.ones(x.shape[0]) / x.shape[0],
            jnp.ones(y.shape[0]) / y.shape[0],
            d_euclidean_matrix(x, y),
            epsilon
        )[0]
    )
    vg_pc2wd = jit(jax.value_and_grad(f_pc2wd, argnums=0, has_aux=False))  # point cloud to wasserstein distance

    costs, source_points_list = [], []
    for _ in range(500):  # optimization loop
        cost, grad = vg_pc2wd(source_points, target_points)
        source_points = source_points - 0.1 * grad
        costs.append(cost)
        source_points_list.append(source_points)

    print(f"calculation finished, wasserstein dist: {costs[-1]}")

    fig, ax = plt.subplots()
    def update_anim(source_tmp):
        ax.cla()
        ax.scatter(source_tmp[:, 0], source_tmp[:, 1], c='b', label="source")
        ax.scatter(source_points_start[:, 0], source_points_start[:, 1], c='g', label="source_start", marker="o")
        ax.scatter(target_points[:, 0], target_points[:, 1], c='r', label="target", marker="x")
        ax.legend()
        return ax

    anim = FuncAnimation(fig, update_anim, frames=source_points_list, interval=20, cache_frame_data=False)
    anim.save("output.gif", writer="pillow")
    plt.close()

