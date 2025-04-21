import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


@jax.jit
def find_surrounding_cube(
    grid_pts: jnp.ndarray,
    mh: float,
    teff: float,
    logg: float,
) -> jnp.ndarray:

    # there's gotta be a better way to do this...
    def single_dimension_vals(coords, val):
        coords = jnp.sort(coords)
        flip = coords <= val
        lower_idx = jnp.sum(flip) - 1
        lower_idx = jnp.clip(lower_idx, 0, len(coords) - 2)
        return jnp.array([coords[lower_idx], coords[lower_idx + 1]])

    mh_vals = single_dimension_vals(grid_pts[:, 0], mh)
    teff_vals = single_dimension_vals(grid_pts[:, 1], teff)
    logg_vals = single_dimension_vals(grid_pts[:, 2], logg)

    cube_vals = jnp.array(
        [
            [mh_vals[0], teff_vals[0], logg_vals[0]],
            [mh_vals[0], teff_vals[0], logg_vals[1]],
            [mh_vals[0], teff_vals[1], logg_vals[0]],
            [mh_vals[0], teff_vals[1], logg_vals[1]],
            [mh_vals[1], teff_vals[0], logg_vals[0]],
            [mh_vals[1], teff_vals[0], logg_vals[1]],
            [mh_vals[1], teff_vals[1], logg_vals[0]],
            [mh_vals[1], teff_vals[1], logg_vals[1]],
        ]
    )
    # jax.debug.print("attempted cube vals: {x}", x=cube_vals)

    scale = jnp.ptp(grid_pts, axis=0)

    def find_cube_idx(cube_val):
        dist = jnp.sum(((grid_pts - cube_val) / scale) ** 2, axis=1)
        return jnp.argmin(dist)

    cube_idxs = jax.vmap(find_cube_idx)(cube_vals)

    return cube_idxs


@jax.jit
def trilinearly_interpolated_weights(cube_points, query_point):
    # this one was thanks to Claude

    # Find min and max coordinates along each dimension
    x_min, y_min, z_min = jnp.min(cube_points, axis=0)
    x_max, y_max, z_max = jnp.max(cube_points, axis=0)

    # Normalize the query point coordinates to [0,1] within the cube
    query_norm = (query_point - jnp.array([x_min, y_min, z_min])) / (
        jnp.array([x_max, y_max, z_max]) - jnp.array([x_min, y_min, z_min])
    )

    # Clip to ensure we stay within [0,1]
    query_norm = jnp.clip(query_norm, 0.0, 1.0)

    # Identify if each vertex is at min (0) or max (1) for each dimension using a tolerance
    tol = 1e-6
    vertex_positions = (
        jnp.abs(cube_points - jnp.expand_dims(jnp.array([x_min, y_min, z_min]), 0))
        > tol
    )
    vertex_positions = vertex_positions.astype(jnp.float32)

    # Calculate the weight for each vertex using vectorized operations
    # For each dimension:
    # - if vertex is at max (1), use query_norm as weight
    # - if vertex is at min (0), use (1-query_norm) as weight
    x_weights = vertex_positions[:, 0] * query_norm[0] + (
        1 - vertex_positions[:, 0]
    ) * (1 - query_norm[0])
    y_weights = vertex_positions[:, 1] * query_norm[1] + (
        1 - vertex_positions[:, 1]
    ) * (1 - query_norm[1])
    z_weights = vertex_positions[:, 2] * query_norm[2] + (
        1 - vertex_positions[:, 2]
    ) * (1 - query_norm[2])

    # Combine weights from all dimensions by multiplication
    weights = x_weights * y_weights * z_weights

    # Normalize weights to ensure they sum to 1
    weights = weights / jnp.sum(weights)
    return weights
