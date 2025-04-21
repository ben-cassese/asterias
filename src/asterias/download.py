import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import os
import requests
import numpy as np

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
    url: str, local_file_name: str, verbose: bool = True, chunk_size: int = 1024
) -> None:
    local_dir = os.path.dirname(local_file_name)
    os.makedirs(local_dir, exist_ok=True)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        desc = "Downloading {}".format(url)
        bool_progress_bar = verbose == 0
        with tqdm(
            desc=desc,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            disable=bool_progress_bar,
        ) as bar:
            with open(local_file_name, "wb") as file:
                for data in response.iter_content(chunk_size=chunk_size):
                    size = file.write(data)
                    bar.update(size)

    except HTTPError as err:
        raise HTTPError(
            "HTTP error occurred: url={}, msg={}".format(err.request.url, err)
        )

    except requests.exceptions.ConnectionError as err:
        raise ConnectionError(
            "Connection error occurred: url={}, msg={}".format(
                err.request.url, "Cannot connect to URL."
            )
        )

    except Exception as err:
        raise err


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
) -> tuple[jnp.ndarray, list]:
    """
    Given limiting values of all stellar parameters, download the relevant ExoTIC-LD files.
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

    mask = (
        (grid[:, 0] >= mh_lower_limit)
        * (grid[:, 0] <= mh_upper_limit)
        * (grid[:, 1] >= teff_lower_limit)
        * (grid[:, 1] <= teff_upper_limit)
        * (grid[:, 2] >= logg_lower_limit)
        * (grid[:, 2] <= logg_upper_limit)
    )
    grid_pts = grid[mask]
    if np.min(grid_pts[:, 0]) > mh_lower_limit:
        warn(
            f"Grid min mh ({np.min(grid_pts[:,0])}) is above supplied lower limit of {mh_lower_limit}. Consider adjusting bounds to match points included in selected stellar grid. Lowest possible mh with this grid is {np.min(grid[:,0])}"
        )
    if np.max(grid_pts[:, 0]) < mh_upper_limit:
        warn(
            f"Grid max mh ({np.max(grid_pts[:,0])}) is below supplied upper limit of {mh_upper_limit}. Consider adjusting bounds to match points included in selected stellar grid. Highest possible mh with this grid is {np.max(grid[:,0])}"
        )
    if np.min(grid_pts[:, 1]) > teff_lower_limit:
        warn(
            f"Grid min teff ({np.min(grid_pts[:,1])}) is above supplied lower limit of {teff_lower_limit}. Consider adjusting bounds to match points included in selected stellar grid. Lowest possible teff with this grid is {np.min(grid[:,1])}"
        )
    if np.max(grid_pts[:, 1]) < teff_upper_limit:
        warn(
            f"Grid max teff ({np.max(grid_pts[:,1])}) is below supplied upper limit of {teff_upper_limit}. Consider adjusting bounds to match points included in selected stellar grid. Highest possible teff with this grid is {np.max(grid[:,1])}"
        )
    if np.min(grid_pts[:, 2]) > logg_lower_limit:
        warn(
            f"Grid min logg ({np.min(grid_pts[:,2])}) is above supplied lower limit of {logg_lower_limit}. Consider adjusting bounds to match points included in selected stellar grid. Lowest possible logg with this grid is {np.min(grid[:,2])}"
        )
    if np.max(grid_pts[:, 2]) < logg_upper_limit:
        warn(
            f"Grid max logg ({np.max(grid_pts[:,2])}) is below supplied upper limit of {logg_upper_limit}. Consider adjusting bounds to match points included in selected stellar grid. Highest possible logg with this grid is {np.max(grid[:,2])}"
        )

    if verbose:
        print(f"Checking for/downloading {len(grid_pts)} files")

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
            download_file(remote_file_path, local_file_path, verbose)
            if verbose:
                print("Downloaded {}.".format(local_file_path))

        return local_file_path

    files = []
    for grid_pt in grid_pts:
        files.append(retrieve_file(mh=grid_pt[0], teff=grid_pt[1], logg=grid_pt[2]))
    return grid_pts, files
