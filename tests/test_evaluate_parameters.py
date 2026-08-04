"""Tests for miice.evaluate_parameters."""

import numpy as np
import pytest
import skimage.io

from miice.evaluate_parameters import (
    _load_image,
    evaluate_segmentation,
    visualize_illumination,
)


def _blob_image(size: int = 100, blob_slice: slice | None = None) -> np.ndarray:
    """Build a synthetic image: dark background with a bright square blob."""
    image = np.zeros((size, size), dtype=np.uint8)
    if blob_slice is None:
        blob_slice = slice(40, 80)
    image[blob_slice, blob_slice] = 200
    return image


class TestLoadImage:
    def test_returns_array_unchanged(self):
        image = _blob_image()
        assert _load_image(image) is image

    def test_loads_from_path(self, tmp_path):
        image = _blob_image()
        path = tmp_path / "blob.tif"
        skimage.io.imsave(path, image, check_contrast=False)

        loaded = _load_image(path)

        np.testing.assert_array_equal(loaded, image)

    def test_loads_from_str_path(self, tmp_path):
        image = _blob_image()
        path = tmp_path / "blob.tif"
        skimage.io.imsave(path, image, check_contrast=False)

        loaded = _load_image(str(path))

        np.testing.assert_array_equal(loaded, image)


class TestEvaluateSegmentation:
    def test_raises_on_empty_images(self):
        with pytest.raises(ValueError, match="at least one channel"):
            evaluate_segmentation({})

    def test_returns_boolean_mask_per_channel(self):
        images = {"ch1": _blob_image(), "ch2": _blob_image()}

        masks = evaluate_segmentation(images)

        assert set(masks) == {"ch1", "ch2"}
        for mask in masks.values():
            assert mask.dtype == bool
            assert mask.shape == (100, 100)
            assert mask[60, 60]  # center of blob
            assert not mask[5, 5]  # far corner background

    def test_accepts_path_and_array_inputs_together(self, tmp_path):
        image = _blob_image()
        path = tmp_path / "blob.tif"
        skimage.io.imsave(path, image, check_contrast=False)
        images = {"from_path": path, "from_array": image}

        masks = evaluate_segmentation(images)

        assert set(masks) == {"from_path", "from_array"}
        np.testing.assert_array_equal(masks["from_path"], masks["from_array"])

    def test_forwards_segment_kwargs(self):
        image = _blob_image()  # large blob, area 1600
        image[5:13, 5:13] = 200  # small isolated speck, area 64

        kept_masks = evaluate_segmentation({"ch1": image}, min_size=10)
        removed_masks = evaluate_segmentation({"ch1": image}, min_size=200)

        assert kept_masks["ch1"][0:20, 0:20].sum() > 0
        assert removed_masks["ch1"][0:20, 0:20].sum() == 0
        # the large blob should survive cleanup in both cases
        assert kept_masks["ch1"][50:70, 50:70].all()
        assert removed_masks["ch1"][50:70, 50:70].all()


class TestVisualizeIllumination:
    def test_returns_the_loaded_image_unchanged(self):
        image = _blob_image()

        returned = visualize_illumination(image)

        np.testing.assert_array_equal(returned, image)

    def test_loads_and_returns_image_from_path(self, tmp_path):
        image = _blob_image()
        path = tmp_path / "blob.tif"
        skimage.io.imsave(path, image, check_contrast=False)

        returned = visualize_illumination(path)

        np.testing.assert_array_equal(returned, image)

    def test_accepts_custom_clip_percentiles(self):
        image = _blob_image()

        returned = visualize_illumination(image, clip_percentiles=(0, 100))

        np.testing.assert_array_equal(returned, image)
