import jax

jax.config.update("jax_enable_x64", True)

import os

import numpy as np
import pytest

# mu sampling, descending from disk centre, matching the layout of a real kurucz file
MUS = np.array(
    [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01]
)
WAVELENGTHS = np.linspace(4000.0, 12000.0, 400)


def true_u_coeffs(mh, teff, logg):
    """
    The quadratic limb darkening coefficients of the synthetic star at a grid point.

    These are deliberately *linear* in the stellar parameters. Trilinear interpolation is
    exact for linear functions, so anything the package reconstructs in between grid points
    can be checked against this closed form rather than against a regression snapshot.
    """
    u1 = 0.40 + 0.10 * mh + 2.0e-5 * (teff - 5000.0) + 0.05 * (logg - 4.5)
    u2 = 0.20 - 0.05 * mh + 1.0e-5 * (teff - 5000.0) - 0.02 * (logg - 4.5)
    return u1, u2


def true_profile(mus, mh, teff, logg):
    """Intensity profile, normalized to 1 at disk centre."""
    u1, u2 = true_u_coeffs(mh, teff, logg)
    return 1.0 - u1 * (1.0 - mus) - u2 * (1.0 - mus) ** 2


def write_synthetic_spectra(path, mh, teff, logg):
    """
    Write one stellar model in the ExoTiC-LD format.

    The intensity is separable, I(lam, mu) = F(lam) * profile(mu). The package divides the
    integrated profile by its value at disk centre, so F(lam) cancels exactly and the
    recovered profile must equal profile(mu) no matter what the filter does. That makes the
    expected answer analytic.
    """
    flux = 1e15 * (WAVELENGTHS / 5000.0) ** -2.0  # arbitrary; must cancel
    profile = true_profile(MUS, mh, teff, logg)
    intensities = flux[:, None] * profile[None, :]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("synthetic test spectra\n")
        f.write(" ".join(str(m) for m in MUS) + "\n")
        for wav, row in zip(WAVELENGTHS, intensities):
            f.write(" ".join(["%.6e" % wav] + ["%.6e" % v for v in row]) + "\n")


def build_grid_on_disk(root, grid_name, mhs, teffs, loggs, skip=()):
    """
    Lay out a synthetic stellar grid exactly where the downloader would cache it.

    download_all_data() skips any file that already exists, so a grid written here is used
    verbatim and no network access happens. `skip` punches holes in the grid, which is how
    the phoenix raggedness is reproduced without touching the real 440 GB of models.
    """
    skip = {(round(m, 2), int(t), round(g, 1)) for m, t, g in skip}
    for mh in mhs:
        for teff in teffs:
            for logg in loggs:
                if (round(mh, 2), int(teff), round(logg, 1)) in skip:
                    continue
                path = os.path.join(
                    root,
                    grid_name,
                    "MH{}".format(round(mh, 2)),
                    "teff{}".format(int(teff)),
                    "logg{}".format(round(logg, 1)),
                    "{}_spectra.dat".format(grid_name),
                )
                write_synthetic_spectra(path, mh, teff, logg)
    return root


# a box that exists in the real kurucz grid, so the real KURUCZ_GRID selects exactly these
KURUCZ_MHS = [-0.1, 0.0, 0.1]
KURUCZ_TEFFS = [4750, 5000, 5250]
KURUCZ_LOGGS = [4.0, 4.5, 5.0]


@pytest.fixture
def synthetic_kurucz(tmp_path):
    """A complete 3x3x3 synthetic kurucz grid, cached where the downloader expects it."""
    return build_grid_on_disk(
        str(tmp_path), "kurucz", KURUCZ_MHS, KURUCZ_TEFFS, KURUCZ_LOGGS
    )


@pytest.fixture
def flat_filter():
    """A filter that transmits uniformly across the synthetic spectra."""
    wavelengths = np.linspace(4000.0, 12000.0, 200)
    throughput = np.ones_like(wavelengths)
    return wavelengths, throughput


@pytest.fixture
def sloped_filter():
    """A filter with real wavelength structure, to prove F(lam) genuinely cancels."""
    wavelengths = np.linspace(4000.0, 12000.0, 200)
    throughput = np.exp(-(((wavelengths - 7000.0) / 2000.0) ** 2))
    return wavelengths, throughput
