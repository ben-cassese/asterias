import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np

from asterias.download import download_all_data
from asterias.interpolate import find_surrounding_cube, trilinearly_interpolated_weights
from asterias.compute import compute_all_profiles


def _get_ldcs(
    mus: jnp.ndarray,
    grid_pts: jnp.ndarray,
    grid_profiles: jnp.ndarray,
    poly_deg: int,
    mh: float,
    teff: float,
    logg: float,
) -> jnp.ndarray:

    cube_idxs = find_surrounding_cube(grid_pts, mh, teff, logg)
    cube_pts = grid_pts[cube_idxs]

    # jax.debug.print("cube idxs: {x}", x=cube_idxs)
    # jax.debug.print("cube pts: {x}", x=cube_pts)

    weights = trilinearly_interpolated_weights(cube_pts, jnp.array([mh, teff, logg]))
    # jax.debug.print("weights: {x}", x=weights)
    vals = grid_profiles[cube_idxs]

    # weighted_profiles = vals[1]
    weighted_profiles = jnp.sum(vals * weights[:, None, None], axis=0)
    weighted_profiles = weighted_profiles / weighted_profiles[:, -1][:, None]

    poly_coeffs = jax.vmap(jnp.polyfit, in_axes=(None, 0, None))(
        mus, weighted_profiles, poly_deg
    )
    return poly_coeffs


class LimbDarkeningCoefficients:

    def __init__(
        self,
        stellar_grid: str,
        mh_lower_limit: float,
        mh_upper_limit: float,
        teff_lower_limit: float,
        teff_upper_limit: float,
        logg_lower_limit: float,
        logg_upper_limit: float,
        wavelength_ranges: np.ndarray,
        filter_wavelengths: np.ndarray,
        filter_throughput: np.ndarray,
        poly_deg: int,
        ld_data_path: str,
        verbose: bool = True,
    ):
        self.poly_deg = poly_deg
        self.grid_pts, self.stellar_files = download_all_data(
            mh_lower_limit=mh_lower_limit,
            mh_upper_limit=mh_upper_limit,
            teff_lower_limit=teff_lower_limit,
            teff_upper_limit=teff_upper_limit,
            logg_lower_limit=logg_lower_limit,
            logg_upper_limit=logg_upper_limit,
            stellar_grid=stellar_grid,
            ld_data_path=ld_data_path,
            verbose=verbose,
        )
        self.mus, self.stellar_profiles, self.dense_mus, self.interpolated_profiles = (
            compute_all_profiles(
                self.stellar_files,
                wavelength_ranges,
                filter_wavelengths,
                filter_throughput,
            )
        )

        self.grid_pts = jnp.array(self.grid_pts)
        self.mus = jnp.array(self.mus)
        self.stellar_profiles = jnp.array(self.stellar_profiles)
        self.dense_mus = jnp.array(self.dense_mus)
        self.interpolated_profiles = jnp.array(self.interpolated_profiles)

        self.get_ldcs = jax.jit(
            jax.tree_util.Partial(
                _get_ldcs,
                self.dense_mus,
                self.grid_pts,
                self.interpolated_profiles,
                self.poly_deg,
            )
        )
