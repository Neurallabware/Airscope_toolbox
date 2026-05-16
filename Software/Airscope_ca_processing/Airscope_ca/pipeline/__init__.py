"""Pipeline stage entry points for calcium-imaging movie processing.

The functions exported here are the supported public API for the staged PICO
calcium extraction workflow. Each stage accepts the composed Hydra config,
shared path object, and pipeline logger used by :mod:`Airscope_ca.process_script`.
"""

from .load_raw_movie import get_frame_count, load_raw_movie
from .motion_correction import correct_motion
from .preprocess_movie import preprocess_movie
from .background_removal import remove_background
from .extract_neural_signals import (
    extract_neural_signals,
    load_extracted_signals,
)
from .export_results import export_calcium_results
