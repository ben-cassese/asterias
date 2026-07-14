import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import os
import time
import requests
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from requests.exceptions import HTTPError, ConnectionError
from warnings import warn
from tqdm import tqdm

from asterias.constants import (
    KURUCZ_GRID,
    PHOENIX_GRID,
    MPS1_GRID,
    MPS2_GRID,
    REMOTE_LD_DATA_PATH,
)


# entirely lifted from ExoTiC-LD/exotic_ld/ld_requests.py
def download_file(
    url: str,
    local_file_name: str,
    verbose: bool = True,
    chunk_size: int = 1024,
    n_retries: int = 5,
) -> None:
    local_dir = os.path.dirname(local_file_name)
    os.makedirs(local_dir, exist_ok=True)

    # the server will drop connections if you hit it with many parallel requests,
    # so back off and retry rather than losing a long download to one refused connection
    for attempt in range(n_retries):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            desc = "Downloading {}".format(url)
            bool_progress_bar = verbose == 0
            # write to a temporary file first, then rename, so that an interrupted
            # download never leaves a truncated file that looks complete on a rerun
            partial_file_name = local_file_name + ".part"
            with tqdm(
                desc=desc,
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
                disable=bool_progress_bar,
            ) as bar:
                with open(partial_file_name, "wb") as file:
                    for data in response.iter_content(chunk_size=chunk_size):
                        size = file.write(data)
                        bar.update(size)
            os.replace(partial_file_name, local_file_name)
            return

        except HTTPError as err:
            # a 404 means the grid point genuinely isn't on the server; retrying won't help
            raise HTTPError(
                "HTTP error occurred: url={}, msg={}".format(err.request.url, err)
            )

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as err:
            if attempt == n_retries - 1:
                raise ConnectionError(
                    "Connection error occurred after {} attempts: url={}, msg={}".format(
                        n_retries, url, "Cannot connect to URL."
                    )
                ) from err
            time.sleep(2**attempt)

        except Exception as err:
            raise err


def _select_block(grid: np.ndarray, bounds: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick the block of grid points needed to interpolate anywhere inside the given bounds.

    Note this *brackets* the bounds rather than selecting the points inside them. To
    interpolate at some teff you need a model on either side of it, so the block has to
    reach outwards to the first grid value below the lower limit and the first above the
    upper limit. Selecting only the points strictly inside the bounds is not enough: a
    well-characterized star has a prior narrower than the grid spacing, which would select
    a single value along an axis and leave nothing to interpolate between.

    Returns the complete rectangular block spanning those bracketing values, together with
    a boolean mask flagging the points the server does not actually have. Only phoenix has
    such holes; for the other grids the mask is all False.
    """
    axis_values = []
    for i, (lo, hi) in enumerate(bounds):
        axis = np.unique(grid[:, i])

        below = np.nonzero(axis <= lo)[0]
        above = np.nonzero(axis >= hi)[0]
        i_lo = below[-1] if len(below) else 0
        i_hi = above[0] if len(above) else len(axis) - 1

        # bounds pinned to a single grid value (e.g. teff_lower == teff_upper) would leave
        # one value on this axis and nothing to interpolate between, so widen by one step
        if i_lo == i_hi:
            if i_hi + 1 < len(axis):
                i_hi += 1
            else:
                i_lo -= 1

        axis_values.append(axis[i_lo : i_hi + 1])

    block = np.array(
        [
            (mh, teff, logg)
            for mh in axis_values[0]
            for teff in axis_values[1]
            for logg in axis_values[2]
        ]
    )
    present = {tuple(pt) for pt in grid}
    missing = np.array([tuple(pt) not in present for pt in block])

    return block, missing


def download_all_data(
    mh_lower_limit: float,
    mh_upper_limit: float,
    teff_lower_limit: float,
    teff_upper_limit: float,
    logg_lower_limit: float,
    logg_upper_limit: float,
    stellar_grid: str,
    ld_data_path: str,
    verbose: bool = True,
    n_workers: int = 1,
) -> tuple[jnp.ndarray, list]:
    """
    Given limiting values of all stellar parameters, download the relevant ExoTIC-LD files.

    Set n_workers > 1 to fetch files concurrently, which is much faster when pulling
    many grid points at once.
    """
    if stellar_grid == "kurucz":
        grid = KURUCZ_GRID
    elif stellar_grid == "phoenix":
        grid = PHOENIX_GRID
    elif stellar_grid == "mps1":
        grid = MPS1_GRID
    elif stellar_grid == "mps2":
        grid = MPS2_GRID
    else:
        raise ValueError("stellar grid not recognized")

    bounds = [
        (mh_lower_limit, mh_upper_limit),
        (teff_lower_limit, teff_upper_limit),
        (logg_lower_limit, logg_upper_limit),
    ]

    # a bound outside the grid's own range cannot be bracketed, so the block stops at the
    # edge and any query out there is clamped to it rather than extrapolated
    for name, (lo, hi), col in zip(["mh", "teff", "logg"], bounds, range(3)):
        g_lo, g_hi = np.min(grid[:, col]), np.max(grid[:, col])
        if lo < g_lo:
            warn(
                f"Requested {name} lower limit of {lo} is below the {stellar_grid} grid, "
                f"which stops at {g_lo}. Queries below {g_lo} will be clamped to it."
            )
        if hi > g_hi:
            warn(
                f"Requested {name} upper limit of {hi} is above the {stellar_grid} grid, "
                f"which stops at {g_hi}. Queries above {g_hi} will be clamped to it."
            )

    grid_pts, missing = _select_block(grid, bounds)

    if missing.any():
        shown = "\n".join(
            "    M_H={}, teff={:.0f}, logg={}".format(*pt)
            for pt in grid_pts[missing][:10]
        )
        if missing.sum() > 10:
            shown += "\n    ... and {} more".format(missing.sum() - 10)
        warn(
            "{} of the {} {} models needed for these bounds are not on the server. Their "
            "profiles will be interpolated from neighbouring grid points:\n{}".format(
                missing.sum(), len(grid_pts), stellar_grid, shown
            )
        )

    if verbose:
        print(f"Checking for/downloading {(~missing).sum()} files")

    # entirely lifted from ExoTiC-LD/exotic_ld/stellar_grids _read_in_stellar_grid
    def retrieve_file(mh, teff, logg):
        mh = 0.0 if mh == -0.0 else mh  # Make zeros not negative.

        local_file_path = os.path.join(
            ld_data_path,
            stellar_grid,
            "MH{}/".format(str(round(mh, 2))),
            "teff{}/".format(int(round(teff))),
            "logg{}/".format(str(round(logg, 1))),
            "{}_spectra.dat".format(stellar_grid),
        )
        remote_file_path = os.path.join(
            REMOTE_LD_DATA_PATH,
            stellar_grid,
            "MH{}".format(str(round(mh, 2))),
            "teff{}".format(int(round(teff))),
            "logg{}".format(str(round(logg, 1))),
            "{}_spectra.dat".format(stellar_grid),
        )

        # Check if exists locally.
        if not os.path.exists(local_file_path):
            # suppress the per-file progress bars when running concurrently,
            # otherwise they interleave into nonsense
            download_file(
                remote_file_path, local_file_path, verbose and (n_workers == 1)
            )
            if verbose and n_workers == 1:
                print("Downloaded {}.".format(local_file_path))

        return local_file_path

    # points the server does not have get None in place of a path; their profiles are
    # filled in from their neighbours once the rest have been read
    def retrieve_or_skip(pt_and_missing):
        pt, is_missing = pt_and_missing
        if is_missing:
            return None
        return retrieve_file(mh=pt[0], teff=pt[1], logg=pt[2])

    work = list(zip(grid_pts, missing))
    if n_workers == 1:
        files = [retrieve_or_skip(w) for w in work]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            files = list(
                tqdm(
                    executor.map(retrieve_or_skip, work),
                    total=len(work),
                    desc="Downloading files",
                    disable=not verbose,
                )
            )

    return grid_pts, files
