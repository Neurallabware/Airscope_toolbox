import os
import shutil
from datetime import datetime
from datetime import timedelta
from pathlib import Path
import sys
import logging
from dataclasses import dataclass
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig, OmegaConf

from Airscope_ca.pipeline import (
    get_frame_count,
    load_raw_movie,
    correct_motion,
    preprocess_movie,
    remove_background,
    extract_neural_signals,
    load_extracted_signals,
    export_calcium_results,
)


@dataclass
class PipelinePaths:
    out_path: str
    mc_out: str
    preprocess_out: str
    rmbg_out: str
    seg_out: str
    mc_zarr_path: str
    preprocess_zarr_path: str
    rmbg_zarr_path: str


def build_paths(cfg) -> PipelinePaths:
    out_path = cfg.out_path
    thresh_pmap = cfg.segmentation.thresh_pmap
    mc_out = os.path.join(out_path, 'mc')
    preprocess_out = os.path.join(out_path, 'preprocess')
    rmbg_out = os.path.join(out_path, 'rmbg')
    seg_out = os.path.join(out_path, f'seg_results_thresh_pmap_{thresh_pmap}')
    paths = PipelinePaths(
        out_path=out_path,
        mc_out=mc_out,
        preprocess_out=preprocess_out,
        rmbg_out=rmbg_out,
        seg_out=seg_out,
        mc_zarr_path=os.path.join(mc_out, 'motion_corrected.zarr'),
        preprocess_zarr_path=os.path.join(preprocess_out, 'video_preprocessed.zarr'),
        rmbg_zarr_path=os.path.join(rmbg_out, 'rmbg.zarr'),
    )
    os.makedirs(mc_out, exist_ok=True)
    os.makedirs(rmbg_out, exist_ok=True)
    os.makedirs(preprocess_out, exist_ok=True)
    os.makedirs(seg_out, exist_ok=True)
    return paths


def setup_logger(out_path):
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in list(root_logger.handlers):
        if getattr(handler, '_airscope_log_file_handler', False):
            root_logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(os.path.join(out_path, 'log_file.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler._airscope_log_file_handler = True
    root_logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = True
    logger.info('Logging INFO and above from root logger to %s', os.path.join(out_path, 'log_file.log'))
    return logger


def _format_duration(seconds):
    return str(timedelta(seconds=round(seconds, 3)))


def _run_timed_stage(stage_times, logger, stage_name, func):
    stage_start = perf_counter()
    logger.info(f'=======>stage start: {stage_name}<=======')
    try:
        result = func()
    except Exception:
        elapsed = perf_counter() - stage_start
        stage_times[stage_name] = elapsed
        logger.exception(
            f'=======>stage failed: {stage_name}; elapsed {_format_duration(elapsed)} ({elapsed:.2f} s)<======='
        )
        raise
    elapsed = perf_counter() - stage_start
    stage_times[stage_name] = elapsed
    logger.info(
        f'=======>stage done: {stage_name}; elapsed {_format_duration(elapsed)} ({elapsed:.2f} s)<======='
    )
    return result


def _log_stage_timing_summary(stage_times, logger):
    logger.info('=======>stage timing summary<=======')
    total_stage_seconds = sum(stage_times.values())
    for stage_name, seconds in stage_times.items():
        logger.info(f'{stage_name}: {_format_duration(seconds)} ({seconds:.2f} s)')
    logger.info(
        f'total measured stage time: {_format_duration(total_stage_seconds)} '
        f'({total_stage_seconds:.2f} s)'
    )


def _cleanup_temporary_chunk_dirs(paths, logger):
    for parent_dir in (paths.mc_out, paths.rmbg_out):
        if not os.path.isdir(parent_dir):
            continue
        for name in os.listdir(parent_dir):
            path = os.path.join(parent_dir, name)
            if name.startswith('chunk_') and os.path.isdir(path):
                shutil.rmtree(path)
                logger.info(f'Removed stale temporary chunk directory: {path}')


def PICO_processing(cfg):
    paths = build_paths(cfg)
    logger = setup_logger(paths.out_path)
    _cleanup_temporary_chunk_dirs(paths, logger)
    stage_times = {}

    logger.info('=======> Arguments <=======')
    start_time = datetime.now()
    pipeline_start = perf_counter()
    logger.info(f'start time: {start_time}')
    logger.info('config:\n%s', OmegaConf.to_yaml(cfg, resolve=True))

    frame_num = get_frame_count(cfg)

    # flow control
    jump_to_rmbg = cfg.jump_to_rmbg
    if jump_to_rmbg:
        jump_to_seg = cfg.jump_to_seg
        jump_to_vis = cfg.jump_to_vis if jump_to_seg else False
    else:
        jump_to_seg = False
        jump_to_vis = False

    if not jump_to_rmbg:
        video = _run_timed_stage(
            stage_times, logger, 'load_raw_movie', lambda: load_raw_movie(cfg, paths, logger)
        )
        video = _run_timed_stage(
            stage_times, logger, 'motion_correction', lambda: correct_motion(cfg, video, paths, logger)
        )
        vessel_img, vessel_mask = _run_timed_stage(
            stage_times, logger, 'preprocess_movie', lambda: preprocess_movie(cfg, video, paths, logger)
        )
    else:
        logger.info('stage skipped: read_frames (jump_to_rmbg=true)')
        logger.info('stage skipped: motion_correction (jump_to_rmbg=true)')
        vessel_img, vessel_mask = _run_timed_stage(
            stage_times, logger, 'load_preprocessed_movie', lambda: preprocess_movie(cfg, None, paths, logger)
        )

    if not jump_to_seg:
        _run_timed_stage(stage_times, logger, 'background_removal', lambda: remove_background(cfg, paths, logger))
    else:
        logger.info('stage skipped: background_removal (jump_to_seg=true)')

    if not jump_to_vis:
        A, C, cm, d1, d2 = _run_timed_stage(
            stage_times, logger, 'extract_neural_signals', lambda: extract_neural_signals(cfg, paths, logger)
        )
    else:
        A, C, cm, d1, d2 = _run_timed_stage(
            stage_times, logger, 'load_extracted_signals', lambda: load_extracted_signals(cfg, paths, logger)
        )

    _run_timed_stage(
        stage_times,
        logger,
        'export_calcium_results',
        lambda: export_calcium_results(cfg, vessel_img, vessel_mask, A, C, cm, d1, d2, paths, logger),
    )

    end_time = datetime.now()
    total_elapsed = perf_counter() - pipeline_start
    logger.info(f'end time: {end_time}')
    _log_stage_timing_summary(stage_times, logger)
    logger.info(f'total pipeline wall time: {_format_duration(total_elapsed)} ({total_elapsed:.2f} s)')
    logger.info(f'{end_time-start_time} spent for {frame_num} calcium processing.')


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    PICO_processing(cfg)


if __name__ == '__main__':
    main()
