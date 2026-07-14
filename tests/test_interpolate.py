import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import itertools

import numpy as np
import pytest

from asterias.interpolate import find_surrounding_cube, trilinearly_interpolated_weights

MHS = np.array([-0.2, -0.1, 0.0, 0.1])
TEFFS = np.array([4500.0, 4750.0, 5000.0])
LOGGS = np.array([4.0, 4.5, 5.0])


def rectangular_grid(mhs=MHS, teffs=TEFFS, loggs=LOGGS):
    a, b, c = np.meshgrid(mhs, teffs, loggs, indexing="ij")
    return jnp.array(np.stack([a.ravel(), b.ravel(), c.ravel()], axis=-1))


def interpolate_point(grid, query):
    """Run the cube lookup and weighting, and return the corners and weights."""
    idxs = find_surrounding_cube(grid, query[0], query[1], query[2])
    corners = grid[idxs]
    weights = trilinearly_interpolated_weights(corners, jnp.array(query))
    return corners, weights


def reconstruct(grid, query):
    """
    Trilinear interpolation is exact for linear functions, and the coordinates themselves
    are linear. So the weighted sum of the cube's corners must return the query point.
    This single identity is what exposes both a collapsed cube and a mis-snapped corner.
    """
    corners, weights = interpolate_point(grid, query)
    return jnp.sum(corners * weights[:, None], axis=0)


class TestWeights:
    def test_weights_sum_to_one_and_are_positive(self):
        grid = rectangular_grid()
        _, w = interpolate_point(grid, (-0.05, 4600.0, 4.2))
        assert jnp.isclose(jnp.sum(w), 1.0)
        assert jnp.all(w >= 0.0)

    def test_interior_point_is_reconstructed(self):
        grid = rectangular_grid()
        query = (-0.05, 4600.0, 4.2)
        assert jnp.allclose(reconstruct(grid, query), jnp.array(query))

    @pytest.mark.parametrize("node", list(itertools.product(MHS, TEFFS, LOGGS)))
    def test_every_grid_node_is_reconstructed_exactly(self, node):
        """
        Regression test: querying at the *upper* edge of any axis used to collapse the
        bracketing cube onto duplicates of a single value, so the weight normalization
        divided by zero and every coefficient came back as nan.
        """
        grid = rectangular_grid()
        out = reconstruct(grid, node)
        assert jnp.all(jnp.isfinite(out)), "nan weights at grid node {}".format(node)
        assert jnp.allclose(out, jnp.array(node))

    def test_upper_corner_specifically(self):
        grid = rectangular_grid()
        corner = (float(MHS[-1]), float(TEFFS[-1]), float(LOGGS[-1]))
        corners, w = interpolate_point(grid, corner)
        assert jnp.all(jnp.isfinite(w))
        # the cube must not degenerate: 8 distinct corners
        assert len({tuple(np.asarray(c)) for c in corners}) == 8

    def test_gradients_are_finite_at_the_upper_edge(self):
        """nan weights at a bound would poison the gradient, not just the value."""
        grid = rectangular_grid()

        def f(mh, teff, logg):
            idxs = find_surrounding_cube(grid, mh, teff, logg)
            w = trilinearly_interpolated_weights(
                grid[idxs], jnp.array([mh, teff, logg])
            )
            return jnp.sum(w * jnp.arange(8))

        grads = jax.jacfwd(f, argnums=(0, 1, 2))(
            float(MHS[-1]), float(TEFFS[-1]), float(LOGGS[-1])
        )
        assert all(jnp.isfinite(g) for g in grads)

    def test_linear_function_is_interpolated_exactly(self):
        """Trilinear interpolation must be exact for any linear function of the params."""
        grid = rectangular_grid()

        def linear(pts):
            return 3.0 + 2.0 * pts[..., 0] + 1e-3 * pts[..., 1] - 0.5 * pts[..., 2]

        rng = np.random.default_rng(0)
        for _ in range(25):
            q = (
                float(rng.uniform(MHS.min(), MHS.max())),
                float(rng.uniform(TEFFS.min(), TEFFS.max())),
                float(rng.uniform(LOGGS.min(), LOGGS.max())),
            )
            corners, w = interpolate_point(grid, q)
            got = jnp.sum(w * linear(corners))
            expected = linear(jnp.array(q))
            assert jnp.allclose(got, expected)

    def test_random_interior_points_reconstruct(self):
        grid = rectangular_grid()
        rng = np.random.default_rng(1)
        for _ in range(50):
            q = (
                float(rng.uniform(MHS.min(), MHS.max())),
                float(rng.uniform(TEFFS.min(), TEFFS.max())),
                float(rng.uniform(LOGGS.min(), LOGGS.max())),
            )
            assert jnp.allclose(reconstruct(grid, q), jnp.array(q))


class TestOutOfBounds:
    def test_queries_outside_the_grid_are_clamped_not_nan(self):
        """
        Documented behaviour: a query outside the grid is clamped to the boundary rather
        than extrapolated. It must at least stay finite -- a sampler that steps out of
        bounds should not poison the chain with nan.
        """
        grid = rectangular_grid()
        for q in [(-5.0, 4600.0, 4.2), (0.0, 99000.0, 4.2), (0.0, 4600.0, 9.9)]:
            _, w = interpolate_point(grid, q)
            assert jnp.all(jnp.isfinite(w))
            assert jnp.isclose(jnp.sum(w), 1.0)

    def test_clamping_returns_the_boundary_value(self):
        grid = rectangular_grid()
        beyond = reconstruct(grid, (0.0, 99000.0, 4.5))
        at_edge = reconstruct(grid, (0.0, float(TEFFS.max()), 4.5))
        assert jnp.allclose(beyond, at_edge)


class TestRaggedGrid:
    def test_missing_corner_corrupts_the_interpolation(self):
        """
        This is *why* asterias refuses ragged blocks up front. With a corner missing, the
        cube lookup silently snaps to the nearest surviving point, the weights still look
        respectable (finite, positive, summing to 1), but the interpolation no longer
        reproduces the query point -- it is quietly blending the wrong models.
        """
        full = rectangular_grid()
        hole = jnp.array([0.0, 5000.0, 4.0])
        ragged = full[~jnp.all(jnp.isclose(full, hole), axis=1)]

        query = (-0.05, 4900.0, 4.2)

        assert jnp.allclose(reconstruct(full, query), jnp.array(query))

        corners, w = interpolate_point(ragged, query)
        assert jnp.isclose(jnp.sum(w), 1.0)  # looks fine...
        assert jnp.all(jnp.isfinite(w))
        # ...but it is wrong, which is exactly the failure the guard exists to prevent
        assert not jnp.allclose(
            jnp.sum(corners * w[:, None], axis=0), jnp.array(query), atol=1e-6
        )
