import numpy as np
import pytest

from asterias.constants import KURUCZ_GRID, PHOENIX_GRID
from asterias.core import _check_wavelength_ranges
from asterias.download import _select_block


def axis_counts(block):
    return [len(np.unique(block[:, i])) for i in range(3)]


class TestBlockSelection:
    """
    The block must BRACKET the requested bounds, not sit inside them. Selecting only the
    points inside the bounds meant a well-characterized star -- whose prior is narrower than
    the grid spacing -- selected a single value along an axis, leaving nothing to
    interpolate between.
    """

    def test_block_brackets_the_bounds(self):
        block, _ = _select_block(KURUCZ_GRID, [(-0.05, 0.05), (4900, 5100), (4.4, 4.6)])
        # every requested bound must lie inside the span of the selected block
        assert block[:, 0].min() <= -0.05 and block[:, 0].max() >= 0.05
        assert block[:, 1].min() <= 4900 and block[:, 1].max() >= 5100
        assert block[:, 2].min() <= 4.4 and block[:, 2].max() >= 4.6

    def test_tight_prior_still_gives_something_to_interpolate_between(self):
        """A Sun-like star with tight priors used to select a degenerate 7x1x1 block."""
        block, _ = _select_block(KURUCZ_GRID, [(-0.3, 0.3), (5622, 5922), (4.14, 4.74)])
        assert min(axis_counts(block)) >= 2

    def test_bounds_pinned_to_a_single_value_are_widened(self):
        """Even lower == upper, exactly on a grid node, must still yield a usable block."""
        block, _ = _select_block(KURUCZ_GRID, [(0.0, 0.0), (5000, 5000), (4.5, 4.5)])
        assert min(axis_counts(block)) >= 2

    def test_block_is_rectangular(self):
        block, _ = _select_block(KURUCZ_GRID, [(-0.3, 0.3), (4500, 5250), (4.0, 5.0)])
        n = axis_counts(block)
        assert len(block) == n[0] * n[1] * n[2]

    def test_bounds_outside_the_grid_clamp_to_its_edge(self):
        block, _ = _select_block(KURUCZ_GRID, [(-9.0, 9.0), (1000, 99000), (0.0, 9.0)])
        assert block[:, 1].min() == KURUCZ_GRID[:, 1].min()
        assert block[:, 1].max() == KURUCZ_GRID[:, 1].max()

    def test_complete_grids_report_no_missing_models(self):
        _, missing = _select_block(KURUCZ_GRID, [(-0.3, 0.3), (4500, 5250), (4.0, 5.0)])
        assert not missing.any()

    def test_phoenix_holes_are_flagged_not_dropped(self):
        """
        The block still spans the full rectangle; the holes are flagged so their profiles
        can be imputed. Losing them from the block would make it non-rectangular again.
        """
        block, missing = _select_block(
            PHOENIX_GRID, [(-0.5, 0.0), (3600, 4100), (3.5, 5.0)]
        )
        n = axis_counts(block)
        assert len(block) == n[0] * n[1] * n[2]
        assert missing.any()
        flagged = {tuple(p) for p in block[missing]}
        assert (0.0, 3700.0, 3.5) in flagged
        assert (0.0, 4000.0, 5.0) in flagged

    def test_flagged_points_are_exactly_those_absent_from_the_grid(self):
        block, missing = _select_block(
            PHOENIX_GRID, [(-1.5, 0.5), (3500, 7000), (3.0, 5.5)]
        )
        real = {tuple(p) for p in PHOENIX_GRID}
        for pt, is_missing in zip(block, missing):
            assert (tuple(pt) not in real) == bool(is_missing)


class TestWavelengthGuard:
    filter_wavelengths = np.linspace(6000.0, 53000.0, 500)
    filter_throughput = np.ones(500)

    def check(self, ranges):
        _check_wavelength_ranges(
            np.array(ranges), self.filter_wavelengths, self.filter_throughput
        )

    def test_ranges_inside_the_filter_pass(self):
        self.check([[6000.0, 7000.0], [7000.0, 8000.0]])

    def test_partial_overlap_passes(self):
        """A bin hanging off the edge of the filter is still integrable."""
        self.check([[4000.0, 7000.0]])

    def test_range_entirely_outside_the_filter_raises(self):
        """This would integrate to zero and hand back a silently all-nan profile."""
        with pytest.raises(ValueError, match="outside the filter"):
            self.check([[1000.0, 2000.0]])

    def test_micron_units_slip_raises(self):
        """The most likely way to trip this: passing microns instead of Angstroms."""
        with pytest.raises(ValueError, match="Angstroms"):
            self.check([[0.6, 0.7], [0.7, 0.8]])

    def test_reversed_range_raises(self):
        with pytest.raises(ValueError, match="lower < upper"):
            self.check([[8000.0, 7000.0]])

    def test_zero_width_range_raises(self):
        """A zero-width bin integrates to zero and yields a flat, unphysical profile."""
        with pytest.raises(ValueError, match="lower < upper"):
            self.check([[7000.0, 7000.0]])
