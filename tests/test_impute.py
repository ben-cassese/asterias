import numpy as np

from asterias.compute import _impute_missing_profiles


def make_block(mhs, teffs, loggs):
    return np.array([(m, t, g) for m in mhs for t in teffs for g in loggs], dtype=float)


def linear_profiles(block):
    """
    A profile that is exactly linear in the stellar parameters.

    The imputation fills a hole by linear interpolation along an axis, so for a linear
    function it must recover the missing profile *exactly*. That turns an approximation
    into something with a closed-form right answer.
    """
    mh, teff, logg = block[:, 0], block[:, 1], block[:, 2]
    base = 1.0 + 0.5 * mh + 1e-4 * (teff - 5000.0) + 0.2 * (logg - 4.5)
    # shape (n_pts, n_bins, n_mus)
    return base[:, None, None] * np.linspace(1.0, 0.4, 10)[None, None, :]


BLOCK = make_block([-0.1, 0.0, 0.1], [4750.0, 5000.0, 5250.0], [4.0, 4.5, 5.0])


class TestImputation:
    def test_exact_for_a_linear_profile_at_an_interior_hole(self):
        truth = linear_profiles(BLOCK)
        # the centre of the block: bracketed along every axis
        hole = np.where((BLOCK == [0.0, 5000.0, 4.5]).all(axis=1))[0][0]

        missing = np.zeros(len(BLOCK), bool)
        missing[hole] = True
        corrupted = truth.copy()
        corrupted[hole] = 0.0

        filled = _impute_missing_profiles(BLOCK, corrupted, missing)
        assert np.allclose(filled[hole], truth[hole])

    def test_untouched_points_are_left_alone(self):
        truth = linear_profiles(BLOCK)
        missing = np.zeros(len(BLOCK), bool)
        missing[13] = True
        corrupted = truth.copy()
        corrupted[13] = 0.0

        filled = _impute_missing_profiles(BLOCK, corrupted, missing)
        keep = ~missing
        assert np.allclose(filled[keep], truth[keep])

    def test_every_interior_hole_is_recovered_exactly(self):
        """Leave-one-out over all points that are bracketed along teff."""
        truth = linear_profiles(BLOCK)
        teffs = np.unique(BLOCK[:, 1])
        interior = [k for k in range(len(BLOCK)) if teffs[0] < BLOCK[k, 1] < teffs[-1]]
        assert interior  # guard against the test vacuously passing

        for k in interior:
            missing = np.zeros(len(BLOCK), bool)
            missing[k] = True
            corrupted = truth.copy()
            corrupted[k] = 0.0
            filled = _impute_missing_profiles(BLOCK, corrupted, missing)
            assert np.allclose(filled[k], truth[k]), "failed at {}".format(BLOCK[k])

    def test_corner_hole_falls_back_and_stays_sane(self):
        """
        A hole on the corner of the block is bracketed along no axis, so it drops to the
        inverse-distance fallback. That cannot be exact, but it must stay finite and land
        within the range spanned by the real models.
        """
        truth = linear_profiles(BLOCK)
        corner = np.where((BLOCK == [-0.1, 4750.0, 4.0]).all(axis=1))[0][0]

        missing = np.zeros(len(BLOCK), bool)
        missing[corner] = True
        corrupted = truth.copy()
        corrupted[corner] = 0.0

        filled = _impute_missing_profiles(BLOCK, corrupted, missing)
        assert np.all(np.isfinite(filled[corner]))
        assert filled[corner].min() >= truth.min() - 1e-9
        assert filled[corner].max() <= truth.max() + 1e-9

    def test_multiple_holes_at_once(self):
        truth = linear_profiles(BLOCK)
        holes = [13, 4]
        missing = np.zeros(len(BLOCK), bool)
        missing[holes] = True
        corrupted = truth.copy()
        corrupted[holes] = 0.0

        filled = _impute_missing_profiles(BLOCK, corrupted, missing)
        assert np.all(np.isfinite(filled))

    def test_uneven_axis_spacing_is_weighted_by_value_not_index(self):
        """
        The grids are not evenly spaced (phoenix teff steps 100 K, then 200 K, then 500 K).
        Interpolating in index space rather than in real units would put the hole in the
        wrong place, so check against a deliberately lopsided axis.
        """
        block = make_block([0.0, 0.1], [5000.0, 5100.0, 5600.0], [4.0, 4.5])
        truth = linear_profiles(block)
        hole = np.where((block == [0.0, 5100.0, 4.0]).all(axis=1))[0][0]

        missing = np.zeros(len(block), bool)
        missing[hole] = True
        corrupted = truth.copy()
        corrupted[hole] = 0.0

        filled = _impute_missing_profiles(block, corrupted, missing)
        # linear truth, so value-weighted interpolation is exact; index-weighted is not
        assert np.allclose(filled[hole], truth[hole])
