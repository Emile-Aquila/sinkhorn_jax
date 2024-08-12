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
        )
    )
    vg_pc2wd = jit(jax.value_and_grad(lambda x, y: f_pc2wd(x, y)[0], argnums=0, has_aux=False))  # point cloud to wasserstein distance
    f_pc2transport = jit(lambda x, y: f_pc2wd(x, y)[1])

    costs, source_points_list, matrixs = [], [], []
    for _ in range(500):  # optimization loop
        cost_tmp, grad = vg_pc2wd(source_points, target_points)
        matrix = f_pc2transport(source_points, target_points)
        source_points = source_points - 0.1 * grad
        costs.append(cost_tmp)
        matrixs.append(matrix)
        source_points_list.append(source_points)

    print(f"calculation finished, wasserstein dist: {costs[-1]}")

    fig, ax, bx = plt.figure(figsize=(12, 7)), plt.subplot(121), plt.subplot(122)

    def update_anim(inputs):
        source_tmp, p, cost = inputs
        ax.cla()
        bx.cla()

        ax.imshow(p)
        ax.set_title("transport matrix")
        ax.set_aspect('equal')

        bx.scatter(source_tmp[:, 0], source_tmp[:, 1], c='b', label="source")
        bx.scatter(source_points_start[:, 0], source_points_start[:, 1], c='g', label="source_start", marker="o")
        bx.scatter(target_points[:, 0], target_points[:, 1], c='r', label="target", marker="x")
        bx.set_title(f"cost: {cost:.3f}")
        bx.set_aspect('equal')

        # ax.legend()
        bx.legend()
        fig.tight_layout()
        return ax, bx

    anim = FuncAnimation(fig, update_anim, frames=zip(source_points_list, matrixs, costs), interval=20, cache_frame_data=False)
    anim.save("output.gif", writer="pillow")
    plt.close()

