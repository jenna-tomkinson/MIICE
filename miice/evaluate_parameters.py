"""Module for previewing segmentation quality and illumination shape on an example
field-of-view (FOV; all channels), to help choose a segmentation/illumination-correction
method before running it across a full directory."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.io

from miice.images import PathLike
from miice.segmentation import segment_individual

logger = logging.getLogger(__name__)


def _load_image(image: PathLike | np.ndarray) -> np.ndarray:
    """Return `image` as an array, loading it from disk first if given a path."""
    if isinstance(image, (str, Path)):
        return skimage.io.imread(image)
    return image


def evaluate_segmentation(
    images: dict[str, PathLike | np.ndarray],
    **segment_kwargs,
) -> dict[str, np.ndarray]:
    """
    Run `segment_individual` on one example image per channel and display the
    resulting mask overlaid on each image, side by side.

    Parameters
    ----------
    images : dict[str, str | Path | numpy.ndarray]
        Mapping of channel label to a single example image for that channel
        (either a path to load or an already-loaded array).
    **segment_kwargs
        Additional keyword arguments forwarded to `segment_individual`
        (e.g. window_size, sigma, min_size, global_factor, k).

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping of channel label to its boolean segmentation mask.
    """
    if not images:
        raise ValueError("`images` must contain at least one channel.")

    loaded = {channel: _load_image(image) for channel, image in images.items()}
    masks = {channel: segment_individual(image, **segment_kwargs) for channel, image in loaded.items()}

    _fig, axes = plt.subplots(1, len(loaded), figsize=(5 * len(loaded), 5))
    if len(loaded) == 1:
        axes = [axes]

    for ax, channel in zip(axes, loaded):
        ax.imshow(loaded[channel], cmap="gray")
        ax.imshow(masks[channel], cmap="magma", alpha=0.4)
        ax.set_title(channel)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return masks


def visualize_illumination(
    image: PathLike | np.ndarray,
    clip_percentiles: tuple[float, float] = (1, 50),
) -> np.ndarray:
    """
    Contrast-stretch the raw image (like manually brightening in Fiji) to
    reveal the large-scale illumination pattern. Foreground objects clip to
    white/black at the extremes rather than smearing into neighboring
    background pixels, so the underlying illumination trend stays visible
    without needing to blur or mask anything out.

    Parameters
    ----------
    image : str | Path | numpy.ndarray
        Image to visualize (a path to load, or an already-loaded array).
    clip_percentiles : tuple[float, float]
        Percentiles of the image used to set the display range, analogous
        to manually brightening/contrast-stretching in Fiji.

    Returns
    -------
    numpy.ndarray
        The (unmodified) image array that was plotted.
    """
    image = _load_image(image)

    vmin, vmax = np.percentile(image, clip_percentiles)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label="Intensity")
    ax.set_title(f"Brightened Image\n(contrast-stretched to {clip_percentiles} percentiles)")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    return image
