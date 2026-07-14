import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np

from asterias.compute import compute_all_profiles, compute_single_profile

from conftest import MUS, true_profile, write_synthetic_spectra


def read_synthetic(tmp_path, mh=0.0, teff=5000, logg=4.5):
    path = str(tmp_path / "spectra.dat")
    write_synthetic_spectra(path, mh, teff, logg)
    return jnp.array(np.loadtxt(path, skiprows=2))


class TestSingleProfile:
    def test_recovers_the_analytic_profile(self, tmp_path, flat_filter):
        """
        The synthetic star is separable, I(lam, mu) = F(lam) * profile(mu), and the profile
        is normalized by its value at disk centre, so F(lam) must cancel exactly. Whatever
        the filter does, the recovered profile has to be the analytic one.
        """
        data = read_synthetic(tmp_path)
        fw, ft = flat_filter
        got = compute_single_profile(
            data, jnp.array([6000.0, 8000.0]), jnp.array(fw), jnp.array(ft)
        )
        # compute_single_profile returns the profile in the file's mu order
        assert jnp.allclose(got, true_profile(MUS, 0.0, 5000, 4.5), atol=1e-6)

    def test_is_independent_of_the_filter_shape(
        self, tmp_path, flat_filter, sloped_filter
    ):
        """A structured throughput must give the same profile as a flat one."""
        data = read_synthetic(tmp_path)
        wr = jnp.array([6000.0, 8000.0])
        flat = compute_single_profile(
            data, wr, jnp.array(flat_filter[0]), jnp.array(flat_filter[1])
        )
        sloped = compute_single_profile(
            data, wr, jnp.array(sloped_filter[0]), jnp.array(sloped_filter[1])
        )
        assert jnp.allclose(flat, sloped, atol=1e-6)

    def test_profile_is_normalized_and_decreasing(self, tmp_path, flat_filter):
        data = read_synthetic(tmp_path)
        fw, ft = flat_filter
        got = compute_single_profile(
            data, jnp.array([6000.0, 8000.0]), jnp.array(fw), jnp.array(ft)
        )
        assert jnp.isclose(got[0], 1.0)  # file order starts at mu = 1
        assert jnp.all(got > 0.0)
        # MUS descends, so intensity must descend too
        assert jnp.all(jnp.diff(got) < 0.0)

    def test_hotter_star_is_more_limb_darkened_here(self, tmp_path, flat_filter):
        """Sanity check that grid parameters actually change the answer."""
        fw, ft = flat_filter
        wr = jnp.array([6000.0, 8000.0])
        cool = compute_single_profile(
            read_synthetic(tmp_path, teff=4750), wr, jnp.array(fw), jnp.array(ft)
        )
        hot = compute_single_profile(
            read_synthetic(tmp_path, teff=5250), wr, jnp.array(fw), jnp.array(ft)
        )
        assert not jnp.allclose(cool, hot)


class TestAllProfiles:
    def test_shapes_and_mu_ordering(self, tmp_path, flat_filter):
        paths = []
        for teff in (4750, 5000):
            p = str(tmp_path / "s{}.dat".format(teff))
            write_synthetic_spectra(p, 0.0, teff, 4.5)
            paths.append(p)

        fw, ft = flat_filter
        wavelength_ranges = jnp.array([[6000.0, 7000.0], [7000.0, 8000.0]])
        mus, profiles, dense_mus, dense_profiles = compute_all_profiles(
            paths, wavelength_ranges, jnp.array(fw), jnp.array(ft)
        )

        assert np.all(np.diff(mus) > 0), "mus must be returned sorted ascending"
        assert profiles.shape == (2, 2, len(MUS))
        assert dense_profiles.shape[:2] == (2, 2)
        assert jnp.all(jnp.isfinite(profiles))
        assert jnp.all(jnp.isfinite(dense_profiles))

    def test_profiles_are_reordered_to_match_sorted_mus(self, tmp_path, flat_filter):
        """
        compute_all_profiles sorts the mu axis. If the intensities were not reordered to
        match, every downstream fit would silently be against a shuffled profile.
        """
        path = str(tmp_path / "s.dat")
        write_synthetic_spectra(path, 0.0, 5000, 4.5)
        fw, ft = flat_filter
        mus, profiles, dense_mus, _ = compute_all_profiles(
            [path], jnp.array([[6000.0, 8000.0]]), jnp.array(fw), jnp.array(ft)
        )
        assert jnp.allclose(
            profiles[0, 0], true_profile(mus, 0.0, 5000, 4.5), atol=1e-6
        )

    def test_dense_interpolation_tracks_the_analytic_profile(
        self, tmp_path, flat_filter
    ):
        path = str(tmp_path / "s.dat")
        write_synthetic_spectra(path, 0.0, 5000, 4.5)
        fw, ft = flat_filter
        _, _, dense_mus, dense_profiles = compute_all_profiles(
            [path], jnp.array([[6000.0, 8000.0]]), jnp.array(fw), jnp.array(ft)
        )
        expected = true_profile(dense_mus, 0.0, 5000, 4.5)
        assert jnp.allclose(dense_profiles[0, 0], expected, atol=1e-4)
