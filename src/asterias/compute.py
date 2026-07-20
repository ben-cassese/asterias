import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
from quadax import simpson
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

    def integrand(waves, stellar_i):
        filter_val = jnp.interp(
            x=waves,
            xp=filter_wavelengths,
            fp=filter_throughput,
            left=0.0,
            right=0.0,
        )
        star_val = jnp.interp(
            x=waves,
            xp=stellar_wavelengths,
            fp=stellar_i,
            left=0.0,
            right=0.0,
        )
        return filter_val * star_val

    # 60,000 is arbitrary but larger than the full number of
    # wavelengths in a phoenix file
    num_pts = 60_000
    integrands = jax.vmap(
        lambda stellar_intensities: integrand(
            waves=jnp.linspace(*wavelength_range, num_pts),
            stellar_i=stellar_intensities,
        )
    )(stellar_intensities.T)
    profile = jax.vmap(
        lambda integrand: simpson(integrand, x=jnp.linspace(*wavelength_range, num_pts))
    )(integrands)
    profile = profile / profile[0]
    return profile


def _impute_missing_profiles(
    grid_pts: np.ndarray, profiles: np.ndarray, missing: np.ndarray
) -> np.ndarray:
    """
    Fill in the profiles of grid points the server does not have.

    Only phoenix needs this: its grid has holes, and a block with a missing corner cannot
    be trilinearly interpolated. Rather than refuse (which would reject about a quarter of
    realistic phoenix stars) the missing profile is reconstructed from its neighbours, once,
    here. Everything downstream then sees a complete rectangular block.

    A hole is filled by linear interpolation along whichever axis has models on both sides
    of it, trying teff first since it is the most finely sampled and the profile varies most
    smoothly along it. If no axis brackets the hole -- it sits on a corner or edge of the
    block -- fall back to an inverse-distance weighted average of the available neighbours.
    """
    profiles = np.array(profiles)
    axes = [np.unique(grid_pts[:, i]) for i in range(3)]
    # index coordinates, so "one step along an axis" is well defined despite uneven spacing
    idx = np.stack(
        [np.searchsorted(axes[i], grid_pts[:, i]) for i in range(3)], axis=-1
    )
    available = ~missing

    for m in np.nonzero(missing)[0]:
        target = idx[m]

        for axis in (1, 2, 0):  # teff, then logg, then mh
            others = [a for a in range(3) if a != axis]
            same_line = np.all(idx[:, others] == target[others], axis=1) & available
            if not same_line.any():
                continue

            pos = idx[same_line, axis]
            below, above = pos[pos < target[axis]], pos[pos > target[axis]]
            if not (len(below) and len(above)):
                continue

            i_lo, i_hi = below.max(), above.min()
            p_lo = profiles[same_line][pos == i_lo][0]
            p_hi = profiles[same_line][pos == i_hi][0]

            # weight by the real axis values, since the grids are not evenly spaced
            v_lo, v_hi = axes[axis][i_lo], axes[axis][i_hi]
            w = (grid_pts[m, axis] - v_lo) / (v_hi - v_lo)
            profiles[m] = (1.0 - w) * p_lo + w * p_hi
            break

        else:
            dist = np.linalg.norm(idx[available] - target, axis=1).astype(float)
            weights = 1.0 / dist**2
            weights /= weights.sum()
            profiles[m] = np.tensordot(weights, profiles[available], axes=1)

    return profiles


def compute_all_profiles(
    filepaths: list,
    wavelength_ranges: jnp.ndarray,
    filter_wavelengths: jnp.ndarray,
    filter_throughput: jnp.ndarray,
    grid_pts: np.ndarray = None,
    poly_order: int = 20,
    stellar_grid: str = None,
    mu_min: float = 0.10,
) -> tuple:
    # entries of filepaths are None where the server has no model for that grid point
    present = [i for i, p in enumerate(filepaths) if p is not None]

    if stellar_grid == "phoenix":
        # phoenix packs its densest mu sampling into a near-limb sliver that lies *above* the
        # photosphere, where the intensity falls to ~0. That cliff makes the downstream global
        # polynomial fit oscillate and go negative, so trim it: read each file's own mu grid,
        # keep mu >= mu_min, and resample onto one shared dense grid. Reading the grid per file
        # also matters because -- unlike the other grids -- phoenix files do not share one mu
        # grid, so the single grid read below would otherwise be misapplied to every file.
        dense_mus = jnp.linspace(mu_min, 1.0, poly_order * 10)
        interpolated_intensities = jnp.zeros(
            (len(filepaths), len(wavelength_ranges), poly_order * 10)
        )

        # can't vmap b/c of the i/o
        for i in present:
            file_mus = np.loadtxt(filepaths[i], skiprows=1, max_rows=1)
            order = np.argsort(file_mus)
            file_mus = file_mus[order]
            keep = file_mus >= mu_min

            stellar_data = np.loadtxt(filepaths[i], skiprows=2)
            grid_pt_intensities = jax.vmap(
                compute_single_profile, in_axes=(None, 0, None, None)
            )(
                stellar_data,
                wavelength_ranges,
                filter_wavelengths,
                filter_throughput,
            )
            interpolator = PchipInterpolator(
                x=file_mus[keep], y=grid_pt_intensities[:, order][:, keep], axis=1
            )
            interpolated_intensities = interpolated_intensities.at[i].set(
                interpolator(dense_mus)
            )

        # phoenix has no single native mu grid, so the dense grid stands in for the "native"
        # outputs too. These two are only returned for inspection; get_ldcs uses the dense pair.
        mus = dense_mus
        actual_intensities = interpolated_intensities
    else:
        mus = np.loadtxt(filepaths[present[0]], skiprows=1, max_rows=1)
        order = np.argsort(mus)
        mus = mus[order]
        dense_mus = jnp.linspace(jnp.min(mus), jnp.max(mus), poly_order * 10)

        actual_intensities = jnp.zeros(
            (len(filepaths), len(wavelength_ranges), len(mus))
        )
        interpolated_intensities = jnp.zeros(
            (len(filepaths), len(wavelength_ranges), poly_order * 10)
        )

        # can't vmap b/c of the i/o
        for i in present:
            stellar_data = np.loadtxt(filepaths[i], skiprows=2)
            grid_pt_intensities = jax.vmap(
                compute_single_profile, in_axes=(None, 0, None, None)
            )(
                stellar_data,
                wavelength_ranges,
                filter_wavelengths,
                filter_throughput,
            )
            interpolator = PchipInterpolator(
                x=mus, y=grid_pt_intensities[:, order], axis=1
            )
            dense_intensities = interpolator(dense_mus)
            actual_intensities = actual_intensities.at[i].set(
                grid_pt_intensities[:, order]
            )
            interpolated_intensities = interpolated_intensities.at[i].set(
                dense_intensities
            )

    missing = np.array([p is None for p in filepaths])
    if missing.any():
        actual_intensities = jnp.array(
            _impute_missing_profiles(grid_pts, actual_intensities, missing)
        )
        interpolated_intensities = jnp.array(
            _impute_missing_profiles(grid_pts, interpolated_intensities, missing)
        )

    # shapes: n_mus, (n_files, n_wavelengths, n_mus), (n_files, n_wavelengths, poly_order*10)
    return mus, actual_intensities, dense_mus, interpolated_intensities
