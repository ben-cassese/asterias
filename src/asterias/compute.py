import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
from quadax import quadgk
from interpax import PchipInterpolator


@jax.jit
def compute_single_profile(
    stellar_data: jnp.ndarray,
    wavelength_range: jnp.ndarray,
    filter_wavelengths: jnp.ndarray,
    filter_throughput: jnp.ndarray,
) -> jnp.ndarray:
    stellar_wavelengths = stellar_data[:, 0]  # shape (n_wavelengths,)
    stellar_intensities = stellar_data[:, 1:]  # shape (n_wavelengths, n_mus)

    def single_mu(stellar_inten):
        def integrand(wav):
            filter_val = jnp.interp(
                x=wav,
                xp=filter_wavelengths,
                fp=filter_throughput,
                left=0.0,
                right=0.0,
            )
            star_val = jnp.interp(
                x=wav,
                xp=stellar_wavelengths,
                fp=stellar_inten,
                left=0.0,
                right=0.0,
            )
            return filter_val * star_val

        y, info = quadgk(
            integrand,
            [wavelength_range[0], wavelength_range[1]],
            epsabs=1e-10,
            epsrel=1e-10,
        )
        return y

    profile = jax.vmap(single_mu, in_axes=(1,))(stellar_intensities)
    profile = profile / profile[0]
    return profile


def compute_all_profiles(
    filepaths: list,
    wavelength_ranges: jnp.ndarray,
    filter_wavelengths: jnp.ndarray,
    filter_throughput: jnp.ndarray,
    poly_order: int = 20,
) -> tuple:
    mus = np.loadtxt(filepaths[0], skiprows=1, max_rows=1)
    order = np.argsort(mus)
    mus = mus[order]
    dense_mus = jnp.linspace(jnp.min(mus), jnp.max(mus), poly_order * 10)

    actual_intensities = jnp.zeros((len(filepaths), len(wavelength_ranges), len(mus)))
    interpolated_intensities = jnp.zeros(
        (len(filepaths), len(wavelength_ranges), poly_order * 10)
    )

    # can't vmap b/c of the i/o
    for i, local_file_path in enumerate(filepaths):
        stellar_data = np.loadtxt(local_file_path, skiprows=2)
        grid_pt_intensities = jax.vmap(
            compute_single_profile, in_axes=(None, 0, None, None)
        )(
            stellar_data,
            wavelength_ranges,
            filter_wavelengths,
            filter_throughput,
        )
        interpolator = PchipInterpolator(x=mus, y=grid_pt_intensities[:, order], axis=1)
        dense_intensities = interpolator(dense_mus)
        actual_intensities = actual_intensities.at[i].set(grid_pt_intensities[:, order])
        interpolated_intensities = interpolated_intensities.at[i].set(dense_intensities)

    # shapes: n_mus, (n_files, n_wavelengths, n_mus), (n_files, n_wavelengths, poly_order*10)
    return mus, actual_intensities, dense_mus, interpolated_intensities
