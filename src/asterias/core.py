import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np

from asterias.download import download_all_data
from asterias.interpolate import find_surrounding_cube, trilinearly_interpolated_weights
from asterias.compute import compute_all_profiles


def _convert_to_ld(poly_coeffs):
    # there's probably an analytic way to do this but this seems to work ok
    order = poly_coeffs.shape[0] - 1
    mus = jnp.linspace(0, 1, order + 1)
    intensities = jnp.polyval(poly_coeffs, mus)
    powers = jnp.arange(order + 1)[1:]
    a = ((1 - mus) ** powers[:, None]).T
    b = intensities - 1
    ld_u_coeffs = -jnp.linalg.lstsq(a, b)[0]
    return ld_u_coeffs


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

    ld_coeffs = jax.vmap(_convert_to_ld)(poly_coeffs)
    return poly_coeffs, ld_coeffs


def _check_wavelength_ranges(
    wavelength_ranges: np.ndarray,
    filter_wavelengths: np.ndarray,
    filter_throughput: np.ndarray,
) -> None:
    """
    Confirm each wavelength bin is well formed and actually sits inside the filter.

    The profile in a bin is the throughput-weighted integral of the stellar spectrum over
    that bin, then normalized by its value at disk center. A bin that misses the filter
    integrates to zero, so the normalization is 0/0 and the whole profile comes back as
    nan with no other complaint. The most likely way to land here is a units slip, e.g.
    passing microns when everything is expected in Angstroms.
    """
    wavelength_ranges = np.atleast_2d(np.asarray(wavelength_ranges))

    bad = wavelength_ranges[wavelength_ranges[:, 0] >= wavelength_ranges[:, 1]]
    if len(bad):
        raise ValueError(
            "Every wavelength range must be [lower, upper] with lower < upper. "
            "Offending ranges: {}".format(bad.tolist())
        )

    # the filter is only meaningfully defined where it actually transmits
    transmits = np.asarray(filter_wavelengths)[np.asarray(filter_throughput) > 0.0]
    f_lo, f_hi = transmits.min(), transmits.max()

    outside = wavelength_ranges[
        (wavelength_ranges[:, 1] <= f_lo) | (wavelength_ranges[:, 0] >= f_hi)
    ]
    if len(outside):
        raise ValueError(
            "The wavelength ranges {} lie entirely outside the filter, which only "
            "transmits between {:.1f} and {:.1f} Angstroms. These bins would integrate "
            "to zero and produce nan coefficients. (Are your wavelengths in Angstroms?)".format(
                outside.tolist(), f_lo, f_hi
            )
        )


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
        n_workers: int = 1,
        mu_min: float = 0.10,
    ):
        self.poly_deg = poly_deg
        _check_wavelength_ranges(
            wavelength_ranges, filter_wavelengths, filter_throughput
        )
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
            n_workers=n_workers,
        )
        self.mus, self.stellar_profiles, self.dense_mus, self.interpolated_profiles = (
            compute_all_profiles(
                self.stellar_files,
                wavelength_ranges,
                filter_wavelengths,
                filter_throughput,
                grid_pts=self.grid_pts,
                stellar_grid=stellar_grid,
                mu_min=mu_min,
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
