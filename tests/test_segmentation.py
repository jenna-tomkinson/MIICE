"""Tests for miice.segmentation."""

import numpy as np
import pytest
import skimage.io

from miice.segmentation import segment_directory, segment_individual


def _blob_image(size: int = 100, blob_slice: slice | None = None) -> np.ndarray:
    """Build a synthetic image: dark background with a bright square blob."""
    image = np.zeros((size, size), dtype=np.uint8)
    if blob_slice is None:
        blob_slice = slice(40, 80)
    image[blob_slice, blob_slice] = 200
    return image


class TestSegmentLocal:
    def test_returns_boolean_mask_matching_shape(self):
        """segment_individual should return a boolean mask with the same shape as the input image."""
        image = _blob_image()
        mask = segment_individual(image)
        assert mask.shape == image.shape
        assert mask.dtype == bool

    def test_marks_bright_blob_as_foreground_and_background_otherwise(self):
        """The bright blob region should be foreground (True) and the dark corner background (False)."""
        image = _blob_image()
        mask = segment_individual(image)
        assert mask[60, 60]  # center of blob
        assert not mask[5, 5]  # far corner background

    def test_constant_image_does_not_raise(self):
        """A uniform image with no contrast should segment without raising an error."""
        image = np.full((64, 64), 128, dtype=np.uint8)
        mask = segment_individual(image)
        assert mask.shape == image.shape
        assert mask.dtype == bool

    def test_min_size_removes_small_isolated_objects(self):
        """Objects smaller than min_size should be dropped while larger objects are kept."""
        image = _blob_image(size=100, blob_slice=slice(40, 80))  # large blob, area 1600
        image[5:13, 5:13] = 200  # small isolated speck, area 64

        kept = segment_individual(image, min_size=10)
        removed = segment_individual(image, min_size=200)

        assert kept[0:20, 0:20].sum() > 0
        assert removed[0:20, 0:20].sum() == 0
        # the large blob should survive cleanup in both cases
        assert kept[50:70, 50:70].all()
        assert removed[50:70, 50:70].all()


class TestSegmentDirectory:
    def _write_images(self, directory, names, size=64):
        """Write a blob test image to `directory` for each filename in `names`."""
        for name in names:
            skimage.io.imsave(directory / name, _blob_image(size=size), check_contrast=False)

    def test_segments_all_matching_files_in_parallel(self, tmp_path):
        """All matching image files in the directory should be segmented, keyed by filename."""
        self._write_images(tmp_path, ["a.tif", "b.tiff", "c.TIF"])

        results = segment_directory(tmp_path, max_workers=2)

        assert len(results) == 3
        assert {p.name for p in results} == {"a.tif", "b.tiff", "c.TIF"}
        for mask in results.values():
            assert mask.dtype == bool
            assert mask.shape == (64, 64)

    def test_ignores_non_matching_extensions_by_default(self, tmp_path):
        """Files with extensions outside the default set (e.g. .txt) should be skipped."""
        self._write_images(tmp_path, ["a.tif"])
        (tmp_path / "notes.txt").write_text("not an image")

        results = segment_directory(tmp_path)

        assert {p.name for p in results} == {"a.tif"}

    def test_extensions_argument_filters_files(self, tmp_path):
        """Passing an explicit extensions argument should restrict which files are processed."""
        skimage.io.imsave(tmp_path / "a.png", _blob_image(size=64), check_contrast=False)
        self._write_images(tmp_path, ["b.tif"])

        results = segment_directory(tmp_path, extensions="png")

        assert {p.name for p in results} == {"a.png"}

    def test_empty_directory_warns_and_returns_empty_dict(self, tmp_path):
        """A directory with no matching images should warn and return an empty results dict."""
        with pytest.warns(UserWarning, match="No images found"):
            results = segment_directory(tmp_path)

        assert results == {}

    def test_forwards_kwargs_to_segment_local(self, tmp_path):
        """Extra keyword arguments (e.g. min_size) should be forwarded to segment_individual."""
        self._write_images(tmp_path, ["a.tif"], size=100)

        results = segment_directory(tmp_path, max_workers=1, min_size=200)

        assert len(results) == 1
