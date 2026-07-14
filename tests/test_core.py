import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import os

import numpy as np
import pytest

from asterias.constants import PHOENIX_GRID
from asterias.core import LimbDarkeningCoefficients
from asterias.download import _select_block

from conftest import (
    KURUCZ_LOGGS,
    KURUCZ_MHS,
    KURUCZ_TEFFS,
    true_profile,
    true_u_coeffs,
    write_synthetic_spectra,
)


@pytest.fixture
def ld(synthetic_kurucz, flat_filter):
    """
    An end-to-end object built entirely from synthetic models cached on disk.

    download_all_data() skips files that already exist, so nothing is fetched over the
    network and the whole pipeline runs against a star whose profile we know in closed form.
    """
    fw, ft = flat_filter
    return LimbDarkeningCoefficients(
        wavelength_ranges=jnp.array([[6000.0, 7000.0], [7000.0, 8000.0]]),
        filter_wavelengths=jnp.array(fw),
        filter_throughput=jnp.array(ft),
        mh_lower_limit=min(KURUCZ_MHS),
        mh_upper_limit=max(KURUCZ_MHS),
        teff_lower_limit=min(KURUCZ_TEFFS),
        teff_upper_limit=max(KURUCZ_TEFFS),
        logg_lower_limit=min(KURUCZ_LOGGS),
        logg_upper_limit=max(KURUCZ_LOGGS),
        stellar_grid="kurucz",
        poly_deg=6,
        ld_data_path=synthetic_kurucz,
        verbose=False,
    )


class TestConstruction:
    def test_selects_the_expected_grid(self, ld):
        assert ld.grid_pts.shape == (27, 3)
        assert len(ld.stellar_files) == 27

    def test_grid_points_line_up_with_their_files(self, ld):
        """
        The profiles are indexed by grid point, so if the file list and the point list ever
        fell out of step every interpolation would silently use the wrong star.
        """
        for pt, path in zip(np.asarray(ld.grid_pts), ld.stellar_files):
            assert "MH{}".format(round(float(pt[0]), 2)) in path
            assert "teff{}".format(int(pt[1])) in path
            assert "logg{}".format(round(float(pt[2]), 1)) in path


class TestGetLdcs:
    def test_recovers_the_true_profile_at_a_grid_node(self, ld):
        mh, teff, logg = 0.0, 5000.0, 4.5
        poly_coeffs, _ = ld.get_ldcs(mh, teff, logg)
        mus = jnp.linspace(0.05, 1.0, 50)
        got = jnp.polyval(poly_coeffs[0], mus)
        expected = true_profile(mus, mh, teff, logg)
        assert jnp.allclose(got, expected, atol=1e-3)

    def test_recovers_the_true_profile_between_grid_nodes(self, ld):
        """
        The synthetic coefficients are linear in the stellar parameters and trilinear
        interpolation is exact for linear functions, so the interpolated profile must match
        the analytic one even away from the nodes.
        """
        mh, teff, logg = 0.05, 4875.0, 4.25
        poly_coeffs, _ = ld.get_ldcs(mh, teff, logg)
        mus = jnp.linspace(0.05, 1.0, 50)
        got = jnp.polyval(poly_coeffs[0], mus)
        expected = true_profile(mus, mh, teff, logg)
        assert jnp.allclose(got, expected, atol=1e-3)

    def test_ld_coefficients_recover_the_quadratic_law(self, ld):
        """
        The synthetic star is exactly quadratic, so in the 1 - c1(1-mu) - c2(1-mu)^2 basis
        the first two coefficients must be u1 and u2 and the rest must be ~0.
        """
        mh, teff, logg = 0.05, 4875.0, 4.25
        _, ld_coeffs = ld.get_ldcs(mh, teff, logg)
        u1, u2 = true_u_coeffs(mh, teff, logg)
        assert jnp.allclose(ld_coeffs[0][0], u1, atol=1e-2)
        assert jnp.allclose(ld_coeffs[0][1], u2, atol=1e-2)
        assert jnp.allclose(ld_coeffs[0][2:], 0.0, atol=1e-2)

    def test_ld_coeffs_are_normalized_to_one_at_disk_centre(self, ld):
        """
        The documented convention: I(mu=1) = 1, NOT unit disk-integrated flux. Transit
        packages renormalize the integral themselves, so changing this would silently
        rescale everyone's profiles.
        """
        _, ld_coeffs = ld.get_ldcs(0.05, 4875.0, 4.25)
        k = jnp.arange(1, ld_coeffs.shape[1] + 1)
        at_centre = 1.0 - jnp.sum(ld_coeffs[0] * (1.0 - 1.0) ** k)
        assert jnp.isclose(at_centre, 1.0)

        # and the disc-integrated flux is therefore NOT 1
        mu = jnp.linspace(0.0, 1.0, 2001)
        profile = 1.0 - jnp.sum(
            ld_coeffs[0][None, :] * (1.0 - mu)[:, None] ** k[None, :], axis=1
        )
        flux = 2 * jnp.pi * jnp.trapezoid(profile * mu, mu)
        assert not jnp.isclose(flux, 1.0)

    def test_both_wavelength_bins_are_returned(self, ld):
        poly_coeffs, ld_coeffs = ld.get_ldcs(0.0, 5000.0, 4.5)
        assert poly_coeffs.shape == (2, 7)
        assert ld_coeffs.shape == (2, 6)

    @pytest.mark.parametrize(
        "node",
        [
            (min(KURUCZ_MHS), min(KURUCZ_TEFFS), min(KURUCZ_LOGGS)),
            (max(KURUCZ_MHS), max(KURUCZ_TEFFS), max(KURUCZ_LOGGS)),
            (0.0, max(KURUCZ_TEFFS), 4.5),
            (max(KURUCZ_MHS), 5000.0, 4.5),
            (0.0, 5000.0, max(KURUCZ_LOGGS)),
        ],
    )
    def test_corners_and_faces_are_finite(self, ld, node):
        """
        Regression test for the collapsed cube: a query sitting exactly on the upper bound
        of any axis used to return nan coefficients, which would silently poison a
        likelihood evaluated at the edge of its prior.
        """
        poly_coeffs, ld_coeffs = ld.get_ldcs(*node)
        assert jnp.all(jnp.isfinite(poly_coeffs))
        assert jnp.all(jnp.isfinite(ld_coeffs))

    def test_upper_corner_matches_the_analytic_profile(self, ld):
        node = (max(KURUCZ_MHS), float(max(KURUCZ_TEFFS)), max(KURUCZ_LOGGS))
        poly_coeffs, _ = ld.get_ldcs(*node)
        mus = jnp.linspace(0.05, 1.0, 50)
        assert jnp.allclose(
            jnp.polyval(poly_coeffs[0], mus), true_profile(mus, *node), atol=1e-3
        )


class TestTransformations:
    def test_is_vmappable(self, ld):
        rng = np.random.default_rng(0)
        n = 32
        mh = jnp.array(rng.uniform(min(KURUCZ_MHS), max(KURUCZ_MHS), n))
        teff = jnp.array(rng.uniform(min(KURUCZ_TEFFS), max(KURUCZ_TEFFS), n))
        logg = jnp.array(rng.uniform(min(KURUCZ_LOGGS), max(KURUCZ_LOGGS), n))

        poly_coeffs, ld_coeffs = jax.vmap(ld.get_ldcs)(mh, teff, logg)
        assert poly_coeffs.shape == (n, 2, 7)
        assert jnp.all(jnp.isfinite(poly_coeffs))
        assert jnp.all(jnp.isfinite(ld_coeffs))

    def test_vmap_agrees_with_looping(self, ld):
        mh = jnp.array([-0.05, 0.0, 0.05])
        teff = jnp.array([4800.0, 5000.0, 5200.0])
        logg = jnp.array([4.1, 4.5, 4.9])
        batched, _ = jax.vmap(ld.get_ldcs)(mh, teff, logg)
        looped = jnp.stack([ld.get_ldcs(m, t, g)[0] for m, t, g in zip(mh, teff, logg)])
        assert jnp.allclose(batched, looped)

    def test_is_differentiable(self, ld):
        jac = jax.jacfwd(ld.get_ldcs, argnums=(0, 1, 2))(0.05, 4875.0, 4.25)
        for output in jac:
            for leaf in output:
                assert jnp.all(jnp.isfinite(leaf))

    def test_gradient_matches_the_analytic_slope(self, ld):
        """
        u1 varies with teff at a known rate, so d(ld_coeffs)/d(teff) has a closed form.
        This checks the derivative is not merely finite but actually right.
        """
        # jacfwd nests as (output, argnum): the ld_coeffs output, differentiated w.r.t. teff
        _, ld_jac = jax.jacfwd(ld.get_ldcs, argnums=(0, 1, 2))(0.0, 4900.0, 4.5)
        d_ld_d_teff = ld_jac[1]
        # u1 = 0.40 + ... + 2.0e-5 * (teff - 5000), so du1/dteff is exactly 2.0e-5
        assert jnp.allclose(d_ld_d_teff[0, 0], 2.0e-5, atol=1e-6)

    def test_gradient_is_finite_at_the_upper_bound(self, ld):
        node = (max(KURUCZ_MHS), float(max(KURUCZ_TEFFS)), max(KURUCZ_LOGGS))
        jac = jax.jacfwd(ld.get_ldcs, argnums=(0, 1, 2))(*node)
        for output in jac:
            for leaf in output:
                assert jnp.all(jnp.isfinite(leaf))


class TestBadInputsAreRejectedBeforeDownloading:
    """
    Each of these must raise during validation, before a single model is fetched. The
    ld_data_path points at an empty directory, so if the guard ever stopped firing the test
    would try to reach the network and fail loudly rather than quietly pass.
    """

    def build(self, tmp_path, flat_filter, **overrides):
        fw, ft = flat_filter
        kwargs = dict(
            wavelength_ranges=jnp.array([[6000.0, 7000.0]]),
            filter_wavelengths=jnp.array(fw),
            filter_throughput=jnp.array(ft),
            mh_lower_limit=-0.1,
            mh_upper_limit=0.1,
            teff_lower_limit=4750,
            teff_upper_limit=5250,
            logg_lower_limit=4.0,
            logg_upper_limit=5.0,
            stellar_grid="kurucz",
            poly_deg=6,
            ld_data_path=str(tmp_path / "empty"),
            verbose=False,
        )
        kwargs.update(overrides)
        return LimbDarkeningCoefficients(**kwargs)

    def test_wavelengths_in_microns_raise(self, tmp_path, flat_filter):
        with pytest.raises(ValueError, match="Angstroms"):
            self.build(tmp_path, flat_filter, wavelength_ranges=jnp.array([[0.6, 0.7]]))

    def test_unknown_grid_raises(self, tmp_path, flat_filter):
        with pytest.raises(ValueError, match="not recognized"):
            self.build(tmp_path, flat_filter, stellar_grid="not_a_grid")


class TestTightPriors:
    def test_a_prior_narrower_than_the_grid_spacing_still_works(
        self, synthetic_kurucz, flat_filter
    ):
        """
        A well-characterized star has a prior narrower than the grid spacing. Selecting only
        the points *inside* the bounds gave a degenerate block and nan coefficients; the
        block must instead bracket the bounds.
        """
        fw, ft = flat_filter
        ld = LimbDarkeningCoefficients(
            wavelength_ranges=jnp.array([[6000.0, 7000.0]]),
            filter_wavelengths=jnp.array(fw),
            filter_throughput=jnp.array(ft),
            mh_lower_limit=-0.01,
            mh_upper_limit=0.01,
            teff_lower_limit=4990,
            teff_upper_limit=5010,
            logg_lower_limit=4.49,
            logg_upper_limit=4.51,
            stellar_grid="kurucz",
            poly_deg=6,
            ld_data_path=synthetic_kurucz,
            verbose=False,
        )
        poly_coeffs, ld_coeffs = ld.get_ldcs(0.0, 5000.0, 4.5)
        assert jnp.all(jnp.isfinite(poly_coeffs))
        assert jnp.all(jnp.isfinite(ld_coeffs))

        mus = jnp.linspace(0.05, 1.0, 50)
        assert jnp.allclose(
            jnp.polyval(poly_coeffs[0], mus),
            true_profile(mus, 0.0, 5000.0, 4.5),
            atol=1e-3,
        )


class TestPhoenixHolesAreImputed:
    def test_a_box_containing_a_real_hole_warns_and_still_works(
        self, tmp_path, flat_filter
    ):
        """
        These bounds straddle two verified holes in the solar phoenix slab. asterias must
        warn, fill them from their neighbours, and return usable coefficients -- roughly a
        quarter of realistic phoenix stars land on a hole, so erroring is not an option.
        """
        bounds = [(-0.5, 0.0), (3600, 4100), (3.5, 5.0)]
        block, missing = _select_block(PHOENIX_GRID, bounds)
        assert missing.any()

        root = str(tmp_path)
        for pt, is_missing in zip(block, missing):
            if is_missing:
                continue  # the server genuinely does not have this one
            write_synthetic_spectra(
                os.path.join(
                    root,
                    "phoenix",
                    "MH{}".format(round(float(pt[0]), 2)),
                    "teff{}".format(int(pt[1])),
                    "logg{}".format(round(float(pt[2]), 1)),
                    "phoenix_spectra.dat",
                ),
                pt[0],
                pt[1],
                pt[2],
            )

        fw, ft = flat_filter
        with pytest.warns(UserWarning, match="interpolated from neighbouring"):
            ld = LimbDarkeningCoefficients(
                wavelength_ranges=jnp.array([[6000.0, 7000.0]]),
                filter_wavelengths=jnp.array(fw),
                filter_throughput=jnp.array(ft),
                mh_lower_limit=bounds[0][0],
                mh_upper_limit=bounds[0][1],
                teff_lower_limit=bounds[1][0],
                teff_upper_limit=bounds[1][1],
                logg_lower_limit=bounds[2][0],
                logg_upper_limit=bounds[2][1],
                stellar_grid="phoenix",
                poly_deg=6,
                ld_data_path=root,
                verbose=False,
            )

        # the imputed block is rectangular, so interpolation works everywhere in the box
        poly_coeffs, ld_coeffs = ld.get_ldcs(-0.25, 3950.0, 4.4)
        assert jnp.all(jnp.isfinite(poly_coeffs))
        assert jnp.all(jnp.isfinite(ld_coeffs))

        # and the synthetic star is linear in the parameters, which the imputation
        # reproduces exactly -- so even a query sitting right on a hole is right
        mus = jnp.linspace(0.05, 1.0, 50)
        on_the_hole, _ = ld.get_ldcs(0.0, 4000.0, 5.0)
        assert jnp.allclose(
            jnp.polyval(on_the_hole[0], mus),
            true_profile(mus, 0.0, 4000.0, 5.0),
            atol=2e-3,
        )
