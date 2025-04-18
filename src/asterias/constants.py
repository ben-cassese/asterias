import numpy as np
import importlib.resources
from asterias import sensitivity_files

REMOTE_LD_DATA_PATH = "https://www.star.bris.ac.uk/exotic-ld-data"

# lifted from ExoTiC-LD/grid_build/generate_tree_pickles.py
# fmt: off
grid_components = {
    "phoenix": {
            "M_H_grid": np.array([-4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.0, 0.5, 1.0]),
            "Teff_grid": np.concatenate([np.arange(2300, 7000, 100), np.arange(7000, 12000, 200), np.arange(12000, 15001, 500)]),
            "logg_grid": np.arange(0.0, 6.01, 0.5)
    },
    "kurucz": {
        "M_H_grid": np.array([-0.1, -0.2, -0.3, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0]),
        "Teff_grid": np.array([3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500]),
        "logg_grid": np.array([4.0, 4.5, 5.0])
    },
    "mps1": {
        "M_H_grid": np.array([-0.1, -0.2, -0.3, -0.4, -0.5, -0.05, -0.6, -0.7, -0.8, -0.9, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75, -0.85, -0.95, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9, -2.0, -2.1, -2.2, -2.3, -2.4, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.05, 0.6, 0.7, 0.8, 0.9, 0.15, 0.25, 0.35, 0.45, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]),
        "Teff_grid": np.arange(3500, 9050, 100),
        "logg_grid": np.array([3.0, 3.5, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.0])
    },
    "mps2": {
        "M_H_grid": np.array([-0.1, -0.2, -0.3, -0.4, -0.5, -0.05, -0.6, -0.7, -0.8, -0.9, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75, -0.85, -0.95, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9, -2.0, -2.1, -2.2, -2.3, -2.4, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.05, 0.6, 0.7, 0.8, 0.9, 0.15, 0.25, 0.35, 0.45, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]),
        "Teff_grid": np.arange(3500, 9050, 100),
        "logg_grid": np.array([3.0, 3.5, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.0])
    },
}
# fmt: on

# equivalent of the generate_stellar_model_points function
_A, _B, _C = np.meshgrid(
    grid_components["phoenix"]["M_H_grid"],
    grid_components["phoenix"]["Teff_grid"],
    grid_components["phoenix"]["logg_grid"],
    indexing="ij",
)
PHOENIX_GRID = np.stack([_A.ravel(), _B.ravel(), _C.ravel()], axis=-1)

_A, _B, _C = np.meshgrid(
    grid_components["kurucz"]["M_H_grid"],
    grid_components["kurucz"]["Teff_grid"],
    grid_components["kurucz"]["logg_grid"],
    indexing="ij",
)
KURUCZ_GRID = np.stack([_A.ravel(), _B.ravel(), _C.ravel()], axis=-1)

_A, _B, _C = np.meshgrid(
    grid_components["mps1"]["M_H_grid"],
    grid_components["mps1"]["Teff_grid"],
    grid_components["mps1"]["logg_grid"],
    indexing="ij",
)
MPS1_GRID = np.stack([_A.ravel(), _B.ravel(), _C.ravel()], axis=-1)

_A, _B, _C = np.meshgrid(
    grid_components["mps2"]["M_H_grid"],
    grid_components["mps2"]["Teff_grid"],
    grid_components["mps2"]["logg_grid"],
    indexing="ij",
)
MPS2_GRID = np.stack([_A.ravel(), _B.ravel(), _C.ravel()], axis=-1)


supported_instruments = [
    "HST_STIS_G430L",
    "HST_STIS_G750L",
    "HST_WFC3_G280p1",
    "HST_WFC3_G280n1",
    "HST_WFC3_G102",
    "HST_WFC3_G141",
    "JWST_NIRSpec_Prism",
    "JWST_NIRSpec_G395H",
    "JWST_NIRSpec_G395M",
    "JWST_NIRSpec_G235H",
    "JWST_NIRSpec_G235M",
    # "JWST_NIRSpec_G140H", # not actually on the ExoTiC-LD server
    "JWST_NIRSpec_G140M-f100",
    "JWST_NIRSpec_G140H-f070",
    "JWST_NIRSpec_G140M-f070",
    "JWST_NIRISS_SOSSo1",
    "JWST_NIRISS_SOSSo2",
    "JWST_NIRCam_F322W2",
    "JWST_NIRCam_F444",
    "JWST_MIRI_LRS",
    "Spitzer_IRAC_Ch1",
    "Spitzer_IRAC_Ch2",
    "TESS",
]

asterias_filters = {}
for s in supported_instruments:
    filename = importlib.resources.files(sensitivity_files).joinpath(
        f"{s}_throughput.csv"
    )
    sensitivity_data = np.loadtxt(filename, skiprows=1, delimiter=",")
    sensitivity_wavelengths = sensitivity_data[:, 0]
    sensitivity_throughputs = sensitivity_data[:, 1]
    asterias_filters[s] = {
        "wavelengths": sensitivity_wavelengths,
        "throughputs": sensitivity_throughputs,
    }
