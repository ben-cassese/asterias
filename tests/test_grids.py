import numpy as np
import pytest

from asterias.constants import (
    KURUCZ_GRID,
    MPS1_GRID,
    MPS2_GRID,
    PHOENIX_GRID,
    asterias_filters,
    supported_instruments,
)

COMPLETE_GRIDS = {"kurucz": KURUCZ_GRID, "mps1": MPS1_GRID, "mps2": MPS2_GRID}


def is_rectangular(grid):
    axes = [np.unique(grid[:, i]) for i in range(3)]
    return len(grid) == len(axes[0]) * len(axes[1]) * len(axes[2])


class TestGridDefinitions:
    @pytest.mark.parametrize("name", sorted(COMPLETE_GRIDS))
    def test_complete_grids_are_rectangular(self, name):
        """kurucz, mps1 and mps2 are fully populated on the server."""
        assert is_rectangular(COMPLETE_GRIDS[name])

    @pytest.mark.parametrize(
        "name,grid",
        sorted(COMPLETE_GRIDS.items()) + [("phoenix", PHOENIX_GRID)],
    )
    def test_no_duplicate_points(self, name, grid):
        assert len({tuple(p) for p in grid}) == len(grid)

    def test_phoenix_matches_what_the_server_actually_serves(self):
        """
        The phoenix grid is NOT the outer product of its axes. Only 5079 of the 10270
        points a rectangular grid would imply are actually served, so the point list is
        shipped verbatim rather than generated. If this number changes, someone has
        regenerated the list and the interpolation guards need rechecking.
        """
        assert PHOENIX_GRID.shape == (5079, 3)

    def test_phoenix_only_has_the_six_served_metallicities(self):
        """M_H = -4.0, -3.0, -2.5 and -2.0 are absent from the server entirely."""
        assert sorted(set(PHOENIX_GRID[:, 0])) == [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]

    def test_phoenix_is_genuinely_ragged(self):
        """
        Guards against someone "simplifying" the phoenix grid back into a meshgrid. If
        this ever passes, the shipped list has been replaced by an outer product and the
        package will start requesting models that 404.
        """
        assert not is_rectangular(PHOENIX_GRID)

    def test_phoenix_known_holes_are_absent(self):
        """Spot-check two holes in the solar slab, verified against the server."""
        pts = {tuple(p) for p in PHOENIX_GRID}
        assert (0.0, 3700.0, 3.5) not in pts
        assert (0.0, 4000.0, 5.0) not in pts
        assert (0.0, 4100.0, 4.5) in pts  # neighbour that does exist


class TestFilters:
    def test_every_supported_instrument_loads(self):
        assert set(asterias_filters) == set(supported_instruments)

    @pytest.mark.parametrize("name", supported_instruments)
    def test_filter_is_well_formed(self, name):
        f = asterias_filters[name]
        wav, tp = f["wavelengths"], f["throughputs"]
        assert len(wav) == len(tp)
        assert np.all(np.isfinite(wav)) and np.all(np.isfinite(tp))
        assert np.all(np.diff(wav) > 0), "wavelengths must be sorted and unique"
        assert np.all(tp >= 0.0)
        assert np.any(tp > 0.0), "filter transmits nowhere"
