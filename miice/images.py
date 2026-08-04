"""Module for matching individual images from raw and corrected directories
and computing intensity slopes from the foreground and background regions."""

import logging
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set up aliases for type hints
PathLike = str | Path
KeyFunc = Callable[[str], str]

logger = logging.getLogger(__name__)

def _normalize_extensions(exts: str | Iterable[str] | None) -> set[str]:
    """
    Normalize an extension string or iterable of extensions into a set of
    lowercase strings that start with a dot (e.g. '.tif', '.tiff').
    If exts is None or empty, return the default TIFF variants.

    Parameters
    ----------
    exts : str | Iterable[str] | None
        A single extension string (e.g. 'tif' or '.tif'), an iterable of such
        strings, or None. Entries may include or omit a leading dot and may
        use any letter case. If None, the function returns the default set.

    Returns
    -------
    set[str]
        A set of normalized extension strings: lowercase and starting with a dot.
        If `exts` is None or if normalization yields no valid entries, returns
        the default set {'.tif', '.tiff'}.

    Raises
    ------
    TypeError
        If any non-str value is present in the provided iterable.
    """
    default = {".tif", ".tiff"}
    if exts is None:
        return default
    if isinstance(exts, str):
        exts = [exts]
    normalized: set[str] = set()
    for e in exts:
        if not isinstance(e, str):
            raise TypeError("extension entries must be str")
        e = e.strip()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        normalized.add(e.lower())
    return normalized or default


def _default_key_func_factory(key_parts: int) -> KeyFunc:
    """
    Return a key function that builds a key from the first `key_parts`
    underscore-separated parts of the filename stem. If there are fewer parts,
    the whole stem is used.

    Parameters
    ----------
    key_parts : int
        Number of underscore-separated components from the filename stem to
        include in the generated key. If `key_parts` is less than 1, the whole
        stem will be used.

    Returns
    -------
    KeyFunc
        A callable that accepts a filename stem (str) and returns the matching
        key (str) formed by joining the first `key_parts` underscore-separated
        components (or the whole stem if there are fewer components).
    """
    def key_func(stem: str) -> str:
        parts = stem.split("_")
        if len(parts) < key_parts:
            return stem
        return "_".join(parts[:key_parts])

    return key_func


def match_image_pairs(
    raw_dir: PathLike,
    corrected_dir: PathLike,
    key_parts: int = 4,
    extensions: str | Iterable[str] | None = None,
    key_func: KeyFunc | None = None,
    *,
    return_unmatched: bool = False,
) -> list[tuple[Path, Path]] | tuple[list[tuple[Path, Path]], list[Path], list[Path]]:
    """
    Match images from two directories based on a common key in their filenames.

    Parameters
    ----------
    raw_dir, corrected_dir : str | Path
        Directories containing raw and corrected images.
    key_parts : int
        Number of underscore-separated parts of the filename stem to use as the key.
        (Used only when `key_func` is not provided.)
    extensions : str | Iterable[str] | None
        File extension(s) to include (e.g., 'tif', '.tif', ['tif','tiff']).
        If None (default), accept common TIFF extensions ('.tif', '.tiff').
    key_func : callable(stem: str) -> str | None
        Optional custom function to produce the matching key from a filename stem.
        If None, a default key function that joins the first `key_parts` underscore
        separated parts is used.
    return_unmatched : bool
        If True, return a tuple (pairs, unmatched_raw, unmatched_corrected) where the
        unmatched lists contain Path objects not paired. Default is False.

    Returns
    -------
    pairs : list of (Path, Path)
        List of matched (raw_image, corrected_image) pairs, sorted by key.
    (optional) unmatched_raw : list of Path
    (optional) unmatched_corrected : list of Path

    Notes
    -----
    - Extensions are matched case-insensitively.
    - If multiple files in one directory map to the same key, the first encountered
      file wins; a warning is emitted. If you need different behavior (e.g. pair
      all combinations), pass a custom key_func where each image set is unique.
    """
    # Convert to Path objects if not already Path instances
    raw_dir = Path(raw_dir)
    corrected_dir = Path(corrected_dir)

    # Normalize extensions to a set of lowercase strings starting with a dot
    exts = _normalize_extensions(extensions)

    # Prepare key function to split filename stems to identify matches
    key_fn = key_func or _default_key_func_factory(key_parts)

    # Gather files in each directory matching the extensions
    raw_files = [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    corrected_files = [p for p in corrected_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]

    # Build maps from keys to file paths
    raw_map: dict[str, Path] = {}
    corrected_map: dict[str, Path] = {}

    # Populate the maps, warning on duplicates
    for p in raw_files:
        key = key_fn(p.stem)
        if key in raw_map:
            warnings.warn(f"Multiple raw files for key '{key}'. Using first: {raw_map[key].name}", UserWarning)
            logger.debug("Duplicate raw key %s -> keeping %s, ignoring %s", key, raw_map[key], p)
            continue
        raw_map[key] = p

    for p in corrected_files:
        key = key_fn(p.stem)
        if key in corrected_map:
            warnings.warn(f"Multiple corrected files for key '{key}'. Using first: {corrected_map[key].name}", UserWarning)
            logger.debug("Duplicate corrected key %s -> keeping %s, ignoring %s", key, corrected_map[key], p)
            continue
        corrected_map[key] = p

    # Find common keys and build pairs
    common_keys = sorted(set(raw_map.keys()) & set(corrected_map.keys()))
    pairs = [(raw_map[k], corrected_map[k]) for k in common_keys]

    # Warn if no matches found
    if not common_keys:
        warnings.warn("No matching image pairs were found.", UserWarning)

    # Warn if some keys did not have matching pairs
    unmatched_raw_keys = set(raw_map.keys()) - set(corrected_map.keys())
    unmatched_corrected_keys = set(corrected_map.keys()) - set(raw_map.keys())
    if unmatched_raw_keys or unmatched_corrected_keys:
        warnings.warn(
            f"Some keys did not have matching pairs: "
            f"{len(unmatched_raw_keys)} unmatched raw, "
            f"{len(unmatched_corrected_keys)} unmatched corrected.",
            UserWarning,
        )

    logger.info("Found %d matched image pairs (of %d raw, %d corrected)", len(pairs), len(raw_files), len(corrected_files))

    # Optionally return unmatched files
    if return_unmatched:
        unmatched_raw = [p for k, p in raw_map.items() if k not in corrected_map]
        unmatched_corrected = [p for k, p in corrected_map.items() if k not in raw_map]
        return pairs, unmatched_raw, unmatched_corrected

    # Return list of matched pairs
    return pairs

def _compute_region_slopes(region: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Compute row- and column-wise average intensities and linear slopes for a region.

    Parameters
    ----------
    region : numpy.ndarray
        2D array containing intensity values for the region of interest. Elements
        that are NaN are ignored when computing averages and fits.

    Returns
    -------
    avg_row : numpy.ndarray
        1D array of mean intensities for each row (NaNs preserved where appropriate).
    avg_col : numpy.ndarray
        1D array of mean intensities for each column (NaNs preserved where appropriate).
    row_slope : float
        Slope of the best-fit line to row means (first-order coefficient). NaN if no valid rows.
    col_slope : float
        Slope of the best-fit line to column means (first-order coefficient). NaN if no valid columns.
    magnitude : float
        Euclidean magnitude of (row_slope, col_slope).

    Notes
    -----
    - Uses numpy.nanmean to ignore NaNs in averages and numpy.polyfit for linear fits.
    - RuntimeWarnings from polyfit on insufficient data are suppressed.
    """
    # Compute average intensities per row and column, ignoring NaNs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg_row = np.nanmean(region, axis=1)
        avg_col = np.nanmean(region, axis=0)

    # Ignore NaNs in rows and columns for slope calculation
    valid_rows = ~np.isnan(avg_row)
    valid_cols = ~np.isnan(avg_col)

    # Compute slopes using numpy.polyfit (linear regression)
    row_slope = np.polyfit(np.arange(avg_row.size)[valid_rows], avg_row[valid_rows], 1)[0] if np.any(valid_rows) else np.nan
    col_slope = np.polyfit(np.arange(avg_col.size)[valid_cols], avg_col[valid_cols], 1)[0] if np.any(valid_cols) else np.nan

    # Compute magnitude of the slope vector for both row and column to get one value
    magnitude = np.sqrt(row_slope**2 + col_slope**2)

    return avg_row, avg_col, row_slope, col_slope, magnitude

def compute_intensity_slopes(image: np.ndarray, mask: np.ndarray) -> tuple[float, float, dict[str, np.ndarray]]:
    """
    Compute row- and column-wise intensity slopes for foreground and background.
    Returns foreground & background slope magnitudes and a dictionary for plotting.
    """
    # Check that image and mask have the same shape
    if image.shape != mask.shape:
        raise ValueError(f"Image shape {image.shape} does not match mask shape {mask.shape}")

    # Perform calculations for foreground
    fg_region = np.where(mask, image, np.nan)
    fg_avg_row, fg_avg_col, fg_row_slope, fg_col_slope, fg_mag = _compute_region_slopes(fg_region)

    # Perform calculations for background
    bg_region = np.where(~mask, image, np.nan)
    bg_avg_row, bg_avg_col, bg_row_slope, bg_col_slope, bg_mag = _compute_region_slopes(bg_region)

    # Compile results into a dictionary for plotting
    results = {
        "fg_avg_row": fg_avg_row,
        "fg_avg_col": fg_avg_col,
        "bg_avg_row": bg_avg_row,
        "bg_avg_col": bg_avg_col,
        "fg_row_slope": fg_row_slope,
        "fg_col_slope": fg_col_slope,
        "bg_row_slope": bg_row_slope,
        "bg_col_slope": bg_col_slope
    }

    return fg_mag, bg_mag, results


def plot_intensity_slopes(results: dict[str, np.ndarray], save_path: Path | None = None) -> None:
    """
    Plot row- and column-wise intensity slopes along with the average intensities per row and column.

    Parameters
    ----------
    results : dict
        Dictionary returned from compute_intensity_slopes. Expected keys:
        "fg_avg_row", "fg_avg_col", "bg_avg_row", "bg_avg_col",
        "fg_row_slope", "fg_col_slope", "bg_row_slope", "bg_col_slope".
    save_path : pathlib.Path or None
        Path to save figure; if None, the figure is shown interactively.

    Returns
    -------
    None
    """
    # Define colors for plotting
    colors = {
        "fg_avg": "#A6CEE3",
        "fg_slope": "#1F78B4",
        "bg_avg": "#FDBF6F",
        "bg_slope": "#FF7F00"
    }

    # Create subplots for rows and columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Rows ---
    x_rows = np.arange(results["fg_avg_row"].size)
    fg_valid_rows = ~np.isnan(results["fg_avg_row"])
    bg_valid_rows = ~np.isnan(results["bg_avg_row"])

    axes[0].plot(results["fg_avg_row"], label="Foreground Avg", color=colors["fg_avg"])
    axes[0].plot(results["bg_avg_row"], label="Background Avg", color=colors["bg_avg"])
    axes[0].plot(
        x_rows[fg_valid_rows],
        results["fg_row_slope"] * x_rows[fg_valid_rows] + np.polyfit(x_rows[fg_valid_rows], results["fg_avg_row"][fg_valid_rows], 1)[1],
        "--", color=colors["fg_slope"], label="Foreground Slope"
    )
    axes[0].plot(
        x_rows[bg_valid_rows],
        results["bg_row_slope"] * x_rows[bg_valid_rows] + np.polyfit(x_rows[bg_valid_rows], results["bg_avg_row"][bg_valid_rows], 1)[1],
        "--", color=colors["bg_slope"], label="Background Slope"
    )
    axes[0].set_xlabel("Row")
    axes[0].set_ylabel("Average Intensity / Slope")
    axes[0].set_title("Row-wise Average Intensity & Slope")

    # --- Columns ---
    x_cols = np.arange(results["fg_avg_col"].size)
    fg_valid_cols = ~np.isnan(results["fg_avg_col"])
    bg_valid_cols = ~np.isnan(results["bg_avg_col"])

    axes[1].plot(results["fg_avg_col"], label="Foreground Avg", color=colors["fg_avg"])
    axes[1].plot(results["bg_avg_col"], label="Background Avg", color=colors["bg_avg"])
    axes[1].plot(
        x_cols[fg_valid_cols],
        results["fg_col_slope"] * x_cols[fg_valid_cols] + np.polyfit(x_cols[fg_valid_cols], results["fg_avg_col"][fg_valid_cols], 1)[1],
        "--", color=colors["fg_slope"], label="Foreground Slope"
    )
    axes[1].plot(
        x_cols[bg_valid_cols],
        results["bg_col_slope"] * x_cols[bg_valid_cols] + np.polyfit(x_cols[bg_valid_cols], results["bg_avg_col"][bg_valid_cols], 1)[1],
        "--", color=colors["bg_slope"], label="Background Slope"
    )
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Average Intensity / Slope")
    axes[1].set_title("Column-wise Average Intensity & Slope")

    # Set unique legend
    handles, labels = [], []
    for ax in axes:
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            if lbl not in labels:
                handles.append(h)
                labels.append(lbl)
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.tight_layout()

    # Show or save the figure
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
