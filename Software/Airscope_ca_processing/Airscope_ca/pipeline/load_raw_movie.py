import glob

from tqdm import tqdm

from Airscope_ca.preprocessing import detect_broken_frame, replace_array
from Airscope_ca.utils.io_videos import (
    count_mp4_frames,
    normalize_data_format,
    read_frames_parallel,
    resolve_mp4_path,
)
from Airscope_ca.Visualization import save_video


def get_frame_count(cfg) -> int:
    """Return the number of raw frames requested for processing.

    The frame count is taken directly from ``cfg.set_frame_num`` when the user
    pins a frame limit. Otherwise, it is discovered from the configured input
    source: MP4 inputs are counted through the video reader, while image
    sequence inputs are counted by matching ``frame_*.{data_format}`` files.

    Args:
        cfg: Composed pipeline configuration. Uses ``set_frame_num``,
            ``data_format``, and ``data_path``.

    Returns:
        Number of raw frames that downstream stages should read.
    """
    if cfg.set_frame_num != 0:
        return cfg.set_frame_num

    data_format = normalize_data_format(cfg.data_format)
    if data_format == "mp4":
        frame_num = count_mp4_frames(cfg.data_path)
    else:
        frame_num = len(glob.glob(f"{cfg.data_path}/frame_*.{data_format}"))
    print(frame_num)
    return frame_num


def load_raw_movie(cfg, paths, logger):
    """Load the raw calcium movie and replace detected broken frames.

    This stage is the canonical entry point from raw acquisition data into the
    processing pipeline. It supports MP4 and image-sequence inputs, reads frames
    in parallel, optionally detects broken frames, and replaces each bad frame
    with the nearest valid frame selected by the preprocessing utilities.

    Args:
        cfg: Composed pipeline configuration. Uses IO worker settings,
            ``data_path``, ``data_format``, ``preprocessing.bad_frame_detect_flag``,
            and debug-video settings.
        paths: Pipeline path bundle. Uses ``out_path`` when writing the optional
            bad-frame replacement debug video.
        logger: Pipeline logger for stage progress and input metadata.

    Returns:
        List-like raw movie frames in grayscale image format, with detected bad
        frames replaced in place.
    """
    frame_num = get_frame_count(cfg)
    data_format = normalize_data_format(cfg.data_format)
    bad_frame_detect_flag = cfg.preprocessing.bad_frame_detect_flag

    flag_array = []
    logger.info('=======>bad frame detection<=======\n')
    if data_format == "mp4":
        logger.info(f'Reading raw MP4 frames from {resolve_mp4_path(cfg.data_path)}')
    else:
        logger.info(
            f'Reading raw {data_format.upper()} frames with {cfg.io_workers} '
            'workers and cv2.IMREAD_GRAYSCALE'
        )
    video = read_frames_parallel(cfg.data_path, frame_num, cfg.io_workers, data_format)

    for i, img in enumerate(tqdm(video)):
        flag = detect_broken_frame(img) if bad_frame_detect_flag else False
        flag_array.append(flag)
        if flag:
            print(f'Broken frame detected at frame_{str(i)}.{data_format}')

    replace_item = replace_array(flag_array)
    print(replace_item)

    for item in replace_item:
        bad_idx, good_idx = item[0], item[1]
        video[bad_idx] = video[good_idx].copy()

    if cfg.save_debug_video:
        save_video(video, cfg.motion.fr, paths.out_path + '/badframe_replaced.avi', quality=cfg.avi_quality)

    return video
