"""Module for generating foreground/background masks from microscopy images using
simple classic segmentation algorithms."""

import logging
import warnings
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import skimage.io
from skimage.filters import gaussian, threshold_otsu, threshold_sauvola
from skimage.morphology import remove_small_holes, remove_small_objects

from miice.images import PathLike, _normalize_extensions

logger = logging.getLogger(__name__)


def segment_individual(
    image: np.ndarray,
    window_size: int = 181,
    sigma: float = 2,
    min_size: int = 100,
    global_factor: float = 0.9,
    k: float = 0.15,
) -> np.ndarray:
    """
    Segment an image into foreground/background using a fast, parameter-light
    local threshold, combined with a global Otsu threshold for robustness.

    This method is intended for quick QC-oriented masks (e.g. for computing
    illumination/contrast metrics) and is not a substitute for a trained
    segmentation model where precise object boundaries matter.

    Parameters
    ----------
    image : numpy.ndarray
        2D grayscale image.
    window_size : int
        Window size (in pixels; must be odd number) used for the local Sauvola threshold.
    sigma : float
        Standard deviation for the Gaussian smoothing applied before thresholding.
    min_size : int
        Maximum size of small objects/holes to remove during mask cleanup.
    global_factor : float
        Multiplier applied to the global Otsu threshold when combining it with
        the local threshold.
    k : float
        Sauvola threshold parameter controlling sensitivity to local contrast.

    Returns
    -------
    numpy.ndarray
        Boolean mask the same shape as `image`, True where foreground.
    """
    # Set the Gaussian smoothing function
    smoothed = gaussian(image, sigma=sigma)

    # Set the local threshold for segmentation
    local_thresh = threshold_sauvola(smoothed, window_size=window_size, k=k)
    local_mask = smoothed > local_thresh

    # Set global threshold for segmentation
    global_thresh = threshold_otsu(smoothed)
    global_mask = smoothed > (global_thresh * global_factor)

    # Combine (intersection-based for robustness)
    mask = (local_mask & (smoothed > 0.8 * global_thresh)) | global_mask

    # Cleanup small objects and holes from being assigned to foreground
    mask = remove_small_objects(mask, max_size=min_size)
    mask = remove_small_holes(mask, max_size=min_size)

    return mask.astype(bool)


def _segment_one(path: Path, segment_kwargs: dict) -> tuple[Path, np.ndarray]:
    """Read an image from disk and segment it. Module-level so it can be pickled
    for use with ProcessPoolExecutor."""
    image = skimage.io.imread(path)
    return path, segment_individual(image, **segment_kwargs)


def segment_directory(
    directory: PathLike,
    extensions: str | Iterable[str] | None = None,
    max_workers: int | None = None,
    **segment_kwargs,
) -> dict[Path, np.ndarray]:
    """
    Segment every image in a directory using `segment_individual`, in parallel.

    Parameters
    ----------
    directory : str | Path
        Directory containing images to segment.
    extensions : str | Iterable[str] | None
        File extension(s) to include (e.g., 'tif', '.tif', ['tif','tiff']).
        If None (default), accept common TIFF extensions ('.tif', '.tiff').
    max_workers : int | None
        Maximum number of worker processes. If None, uses the default for
        `concurrent.futures.ProcessPoolExecutor` (typically the number of CPUs).
    **segment_kwargs
        Additional keyword arguments forwarded to `segment_local`
        (e.g. window_size, sigma, min_size, global_factor, k).

    Returns
    -------
    dict[Path, numpy.ndarray]
        Mapping of image path to its boolean foreground/background mask.
    """
    # Set directory path
    directory = Path(directory)
    # Only detect images after normalization of the extensions
    exts = _normalize_extensions(extensions)
    # Detect all image files
    image_files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts)

    # Warn user if no images are in the directory and return empty results
    if not image_files:
        warnings.warn(f"No images found in {directory} with extensions {sorted(exts)}.", UserWarning)
        return {}

    # Set results as a dictionary with the path to image and the mask (numpy array)
    results: dict[Path, np.ndarray] = {}
    # Parallelize the segmentation across images
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_segment_one, p, segment_kwargs): p for p in image_files}
        for future in as_completed(futures):
            path, mask = future.result()
            results[path] = mask

    # Log results
    logger.info("Segmented %d of %d images in %s", len(results), len(image_files), directory)

    return results
