# asterias

Differentiable, physically-motivated stellar limb darkening for exoplanet transit fits.

When you fit a transit, you have to say something about how the host star's brightness falls
off toward its limb. The usual approach is to sample the coefficients of a parameterized limb
darkening law directly -- for example, the `q1`/`q2` reparameterization of the quadratic law
from [Kipping (2013)](https://scixplorer.org/abs/2013MNRAS.435.2152K/abstract). That works, but
the coefficients are not themselves physical quantities, and it is awkward to fold in what you
already know about the star.

`asterias` lets you sample the *stellar* parameters instead -- effective temperature,
surface gravity, and metallicity -- and turns them into a limb darkening profile on the fly by
interpolating between profiles precomputed from a grid of stellar atmosphere models. Because
the whole path from `(M_H, teff, logg)` to limb darkening coefficients is written in
[JAX](https://github.com/google/jax), it is differentiable, jit-able, and vmap-able, so it drops
into a gradient-based sampler (NUTS/HMC) without any special handling. You can put a prior on
`teff` from spectroscopy and let the limb darkening follow from it.

The method is described in section 3.3 of
[Cassese et al. (2026)](https://scixplorer.org/abs/2026AJ....171..150C/abstract).

## Installation

```bash
pip install asterias
```

## Quickstart

The stellar models themselves are hosted by the
[ExoTiC-LD](https://github.com/Exo-TiC/ExoTiC-LD) project and are downloaded on demand the first
time you ask for a region of the grid. They are cached in `ld_data_path`, so you pay the download
cost once.

Start with the `kurucz` grid: it is by far the smallest (741 models, ~300 MB for the whole
thing), so it is the quickest way to see the package work.

```python
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from asterias import LimbDarkeningCoefficients, asterias_filters

prism = asterias_filters["JWST_NIRSpec_Prism"]

ld = LimbDarkeningCoefficients(
    # the wavelength bins (in Angstroms) you want coefficients for
    wavelength_ranges=jnp.array([[6000.0, 7000.0], [7000.0, 8000.0]]),
    # the instrument throughput the models are integrated against
    filter_wavelengths=jnp.array(prism["wavelengths"]),
    filter_throughput=jnp.array(prism["throughputs"]),
    # the region of stellar parameter space your sampler will explore
    mh_lower_limit=-0.3,
    mh_upper_limit=0.3,
    teff_lower_limit=4500,
    teff_upper_limit=5250,
    logg_lower_limit=4.0,
    logg_upper_limit=5.0,
    stellar_grid="kurucz",
    poly_deg=6,
    ld_data_path="data",
    n_workers=8,  # download the models in parallel
    verbose=True,
)

# for any star inside those bounds, get its limb darkening coefficients
poly_coeffs, ld_coeffs = ld.get_ldcs(mh=0.1, teff=4900.0, logg=4.4)
```

You give the constructor *bounds*, not a single star. It downloads the models needed to bracket
that box up front, so the interpolation afterwards is pure JAX and touches no disk. Set the bounds
to the range your sampler will actually explore -- your priors, essentially -- and no wider, since
the bounds are what drive the download. They can be narrower than the grid spacing; `asterias`
reaches outwards to the surrounding models regardless.

`get_ldcs` returns two things, one row per wavelength bin:

- `poly_coeffs`: coefficients of a degree-`poly_deg` polynomial in `mu`, ready for `jnp.polyval`.
- `ld_coeffs`: the same profile in the `1 - c1*(1-mu) - c2*(1-mu)^2 - ...` convention used by
  packages like [squishyplanet](https://github.com/ben-cassese/squishyplanet) and
  [jaxoplanet](https://github.com/exoplanet-dev/jaxoplanet).

### A note on normalization

Both are normalized so the intensity is **1 at disk centre**, `I(mu=1) = 1`. They are *not*
normalized to unit disk-integrated flux -- integrating either over the stellar disc gives roughly
2.5, not 1.

This is almost always what you want, because transit packages renormalize internally: hand
`ld_coeffs` to `squishyplanet` or `jaxoplanet` and they rescale the profile so the integrated flux
is 1. The overall scale is absorbed by your out-of-transit baseline anyway, so it does not affect
the fit. You do not need to do anything.

The one wrinkle: `ld_coeffs` satisfies `I(mu=1) = 1` *exactly*, since every term of
`1 - c1*(1-mu) - ...` vanishes at `mu = 1`. `poly_coeffs` is off by around 0.2%, because a
least-squares polynomial fit is not constrained to pass through the disc-centre point. If you use
`poly_coeffs` directly as an intensity profile rather than feeding it to a transit package, divide
by `jnp.polyval(poly_coeffs[i], 1.0)` first.

It is differentiable and vectorizable, which is the whole point:

```python
# gradients with respect to the stellar parameters
jax.jacfwd(ld.get_ldcs, argnums=(0, 1, 2))(0.1, 4900.0, 4.4)

# or evaluate a whole chain of posterior draws at once
mh, teff, logg = ...  # each shape (n_samples,)
poly_coeffs, ld_coeffs = jax.vmap(ld.get_ldcs)(mh, teff, logg)
```

## Choosing the filter

`asterias` never assumes an instrument. The only thing it needs is a throughput curve: two
matching arrays, `filter_wavelengths` (in Angstroms, ascending) and `filter_throughput` (any
positive scale -- it is normalized away). The stellar spectrum is integrated against that curve
inside each of your `wavelength_ranges`.

Throughput curves for 23 common instrument modes are bundled, so the common case is a lookup:

```python
from asterias import asterias_filters

print(sorted(asterias_filters))  # what's available
prism = asterias_filters["JWST_NIRSpec_Prism"]
prism["wavelengths"], prism["throughputs"]
```

<details>
<summary>The 23 bundled modes</summary>

`HST_STIS_G430L`, `HST_STIS_G750L`, `HST_WFC3_G102`, `HST_WFC3_G141`, `HST_WFC3_G280n1`,
`HST_WFC3_G280p1`, `JWST_MIRI_LRS`, `JWST_NIRCam_F322W2`, `JWST_NIRCam_F444`,
`JWST_NIRISS_SOSSo1`, `JWST_NIRISS_SOSSo2`, `JWST_NIRSpec_G140H-f070`,
`JWST_NIRSpec_G140H-f100`, `JWST_NIRSpec_G140M-f070`, `JWST_NIRSpec_G140M-f100`,
`JWST_NIRSpec_G235H`, `JWST_NIRSpec_G235M`, `JWST_NIRSpec_G395H`, `JWST_NIRSpec_G395M`,
`JWST_NIRSpec_Prism`, `Spitzer_IRAC_Ch1`, `Spitzer_IRAC_Ch2`, `TESS`

These are the throughput files distributed with ExoTiC-LD.
</details>

But nothing is special about them -- pass any two arrays you like. A ground-based filter, a mode
that isn't bundled, a curve you measured yourself, or a flat top-hat if you want a plain
box-averaged profile:

```python
import numpy as np

# a custom curve read from disk
curve = np.genfromtxt("my_filter.csv", delimiter=",", names=True)
ld = LimbDarkeningCoefficients(
    filter_wavelengths=jnp.array(curve["wave"]),      # Angstroms
    filter_throughput=jnp.array(curve["throughput"]),
    ...
)

# or a top-hat: no instrument weighting at all
waves = jnp.linspace(6000.0, 9000.0, 1000)
ld = LimbDarkeningCoefficients(
    filter_wavelengths=waves,
    filter_throughput=jnp.ones_like(waves),
    ...
)
```

`wavelength_ranges` and `filter_wavelengths` must be in the same units (Angstroms). If a
requested bin falls outside the filter's support the profile is undefined, so `asterias` raises
rather than handing back silent `nan`s -- which is usually how you find out you passed microns.

## Available stellar grids

Set `stellar_grid` to one of the following.

| grid | models | full grid | notes |
|---|---|---|---|
| `kurucz` | 741 | ~0.3 GB | Smallest. Best starting point. |
| `mps1` | 34,160 | ~19 GB | [MPS-ATLAS](https://scixplorer.org/abs/2023RNAAS...7...39K/abstract), solar-scaled abundances. |
| `mps2` | 34,160 | ~19 GB | MPS-ATLAS, alpha-enhanced abundances. |
| `phoenix` | 5,079 | ~440 GB | Files are ~86 MB each. |

### You almost certainly do not need the whole grid

Those "full grid" numbers are worst cases, and they are not what a normal fit costs.
`asterias` only downloads the models needed to bracket the bounds you ask for, so a fit of a
**single star** pulls a small box, not a grid. For a Sun-like star (teff 5772 K, logg 4.44,
`[M/H]` 0.0):

| prior width | `kurucz` | `phoenix` | `mps1` / `mps2` |
|---|---|---|---|
| teff +/-100 K, logg +/-0.2, `[M/H]` +/-0.1 | 27 files, 12 MB | 36 files, 3.0 GB | 120 files, 68 MB |
| teff +/-150 K, logg +/-0.3, `[M/H]` +/-0.3 | 63 files, 27 MB | 45 files, 3.8 GB | 520 files, 296 MB |
| teff +/-300 K, logg +/-0.5, `[M/H]` +/-0.5 | 135 files, 58 MB | 96 files, 8.1 GB | 1512 files, 862 MB |

So a typical one-star fit is tens of megabytes on `kurucz` and a few hundred on `mps1`. Only
`phoenix` stays painful, because each individual model is ~86 MB -- there is no way around that
short of not using `phoenix`. The download is cached in `ld_data_path` and reused, and it is the
*bounds* that drive the cost, not how long you sample for. Widen them only as far as your
sampler will actually wander.

If you genuinely do want a whole grid locally (say, on a cluster), pass `n_workers` to fetch many
files at once. Downloads are resumable: files already on disk are skipped and interrupted
downloads are never left behind as truncated files, so you can just rerun the same call.

### A caveat on `phoenix`: it has holes

Unlike the others, `phoenix` is **not** a complete rectangular grid. Only 6 of its 10
metallicities are actually served (`-1.5, -1.0, -0.5, 0.0, 0.5, 1.0`), and within each of those
there are gaps in the (`teff`, `logg`) plane. Of the 10,270 points a rectangular grid would
contain, only 5,079 exist. The gaps are sparse near solar metallicity and common at low
metallicity.

This matters because trilinear interpolation needs all 8 corners of the cube around your star,
and roughly a quarter of realistic single-star boxes contain at least one hole. Refusing all of
those would make `phoenix` unusable, so instead `asterias` reconstructs a missing model's profile
from its neighbours -- linearly along whichever axis brackets it, preferring `teff` since it is
the most finely sampled -- and warns, naming every model it filled in:

```
UserWarning: 2 of the 48 phoenix models needed for these bounds are not on the server.
Their profiles will be interpolated from neighbouring grid points:
    M_H=0.0, teff=3700, logg=3.5
    M_H=0.0, teff=4000, logg=5.0
```

Filling happens once, up front, so the sampling path stays exact and differentiable. The cost is
accuracy at the filled point. Measured by leave-one-out against real `kurucz` models, an imputed
profile is off by about 1e-3 fractionally, roughly 20% of the difference between two adjacent
grid points. That is small, but it is not nothing: if your star sits right on top of a hole and
you need the last decimal place, prefer a grid without one.

## How this compares to ExoTiC-LD

[ExoTiC-LD](https://github.com/Exo-TiC/ExoTiC-LD)
([Grant & Wakeford 2024](https://scixplorer.org/abs/2024JOSS....9.6816G/abstract)) solves a
related but different problem, and `asterias` leans on it heavily -- the stellar grids, the file
formats, and the download logic all come from ExoTiC-LD, and the instrument throughputs shipped
here are theirs.

The difference is *when* the stellar models enter your fit.

ExoTiC-LD is a preprocessing step. You hand it your best estimate of the star's parameters, and
it hands back the coefficients of a chosen limb darkening law (linear, quadratic, three- and
four-parameter nonlinear, and so on). You then hold those coefficients fixed in your transit fit,
or put a prior on them. If your star's parameters are uncertain, that uncertainty is not
naturally propagated into the transit parameters.

`asterias` is meant to live *inside* the likelihood. The stellar parameters stay free, and the
limb darkening profile is regenerated from the model grid at every evaluation. Uncertainty in
`teff`, `logg`, and `M_H` propagates into the transit posterior automatically, and correlations
between (say) stellar temperature and planet radius show up in the chains rather than being
assumed away. That requires the grid interpolation to be differentiable and fast, which is what
this package provides and what ExoTiC-LD, being a scipy-based preprocessing tool, does not aim for.

The other, smaller difference: `asterias` does not ask you to pick a limb darkening *law*. It
fits a polynomial in `mu` directly to the interpolated intensity profile, so you are not
committing to the quadratic or four-parameter form.

Use ExoTiC-LD if you want coefficients for a standard law and are happy to fix them. Use
`asterias` if you want to marginalize over the star.

## Development

```bash
uv sync --group dev
uv run pytest
```

The tests build synthetic stellar models on disk in the layout the downloader caches into, so
the whole suite runs offline -- nothing is fetched from the ExoTiC-LD server. The synthetic star
has an intensity profile that is separable in wavelength and exactly quadratic in `mu`, with
coefficients that vary linearly across the grid. Trilinear interpolation is exact for linear
functions, so the profile the package reconstructs anywhere in the box can be checked against a
closed form rather than against a stored snapshot.

## Citation

If you use this package, please cite
[Cassese et al. (2026)](https://scixplorer.org/abs/2026AJ....171..150C/abstract), and please also
cite [ExoTiC-LD](https://scixplorer.org/abs/2024JOSS....9.6816G/abstract) along with the
appropriate paper for whichever stellar grid you used, since the underlying models are theirs. We thank the maintainers of ExoTiC-LD for making their code and grids available; `asterias` is entirely built on top of their work.
