"""Local defaults for standalone registration."""

from copy import deepcopy


_REGISTRATION_DEFAULTS = {
    "align_by_chan2": False,
    "nimg_init": 400,
    "maxregshift": 0.1,
    "do_bidiphase": False,
    "bidiphase": 0.0,
    "batch_size": 100,
    "nonrigid": True,
    "maxregshiftNR": 5,
    "block_size": (128, 128),
    "smooth_sigma_time": 0,
    "smooth_sigma": 1.15,
    "spatial_taper": 3.45,
    "th_badframes": 1.0,
    "norm_frames": True,
    "snr_thresh": 1.2,
    "subpixel": 10,
    "two_step_registration": False,
    "reg_tif": False,
    "reg_tif_chan2": False,
}


def default_settings():
    """Return registration defaults in both flat and nested forms.

    Flat keys preserve compatibility with registration functions that index
    settings directly (for example settings["batch_size"]). The nested
    `registration` key preserves compatibility with call sites expecting
    default_settings()["registration"].
    """
    reg = deepcopy(_REGISTRATION_DEFAULTS)
    out = deepcopy(reg)
    out["registration"] = deepcopy(reg)
    return out
