"""Tests for miice.images."""

import numpy as np
import pytest

from miice.images import (
    _compute_region_slopes,
    _default_key_func_factory,
    _normalize_extensions,
    compute_intensity_slopes,
    match_image_pairs,
    plot_intensity_slopes,
)


class TestNormalizeExtensions:
    def test_none_returns_default_tiff_extensions(self):
        assert _normalize_extensions(None) == {".tif", ".tiff"}

    def test_single_string_without_dot_gets_dot_prefixed(self):
        assert _normalize_extensions("png") == {".png"}

    def test_single_string_with_dot_and_mixed_case_is_lowercased(self):
        assert _normalize_extensions(".PNG") == {".png"}

    def test_iterable_of_extensions_is_normalized(self):
        assert _normalize_extensions(["tif", ".JPG", "Png"]) == {".tif", ".jpg", ".png"}

    def test_blank_and_empty_entries_are_skipped(self):
        assert _normalize_extensions(["", "  ", "tif"]) == {".tif"}

    def test_only_blank_entries_falls_back_to_default(self):
        assert _normalize_extensions(["", "  "]) == {".tif", ".tiff"}

    def test_non_str_entry_raises_type_error(self):
        with pytest.raises(TypeError):
            _normalize_extensions([".tif", 5])


class TestDefaultKeyFuncFactory:
    def test_joins_first_n_underscore_parts(self):
        key_func = _default_key_func_factory(2)
        assert key_func("r15_c03_f01_p01") == "r15_c03"

    def test_returns_whole_stem_when_fewer_parts_than_requested(self):
        key_func = _default_key_func_factory(5)
        assert key_func("r15_c03_f01") == "r15_c03_f01"

    def test_whole_stem_used_when_no_underscores(self):
        key_func = _default_key_func_factory(3)
        assert key_func("singlepart") == "singlepart"


class TestMatchImagePairs:
    def _touch(self, directory, names):
        for name in names:
            (directory / name).touch()

    def _dirs(self, tmp_path):
        raw_dir, corrected_dir = tmp_path / "raw", tmp_path / "corrected"
        raw_dir.mkdir()
        corrected_dir.mkdir()
        return raw_dir, corrected_dir

    def test_matches_pairs_by_default_key(self, tmp_path):
        raw_dir, corrected_dir = self._dirs(tmp_path)
        self._touch(raw_dir, ["sample1_wellA1_field1_z1_raw.tif"])
        self._touch(corrected_dir, ["sample1_wellA1_field1_z1_corrected.tif"])

        pairs = match_image_pairs(raw_dir, corrected_dir)

        assert len(pairs) == 1
        assert pairs[0][0].name == "sample1_wellA1_field1_z1_raw.tif"
        assert pairs[0][1].name == "sample1_wellA1_field1_z1_corrected.tif"

    def test_extensions_argument_filters_files(self, tmp_path):
        raw_dir, corrected_dir = self._dirs(tmp_path)
        self._touch(raw_dir, ["a_b_c_d.png", "a_b_c_d.tif"])
        self._touch(corrected_dir, ["a_b_c_d.png", "a_b_c_d.tif"])

        pairs = match_image_pairs(raw_dir, corrected_dir, extensions="png")

        assert len(pairs) == 1
        assert pairs[0][0].suffix == ".png"

    def test_case_insensitive_extension_matching(self, tmp_path):
        raw_dir, corrected_dir = self._dirs(tmp_path)
        self._touch(raw_dir, ["a_1_1_1_raw.TIF"])
        self._touch(corrected_dir, ["a_1_1_1_corrected.tif"])

        pairs = match_image_pairs(raw_dir, corrected_dir)

        assert len(pairs) == 1

    def test_custom_key_func_is_used(self, tmp_path):
        raw_dir, corrected_dir = self._dirs(tmp_path)
        self._touch(raw_dir, ["sample1_raw.tif"])
        self._touch(corrected_dir, ["sample1_corrected.tif"])

        pairs = match_image_pairs(raw_dir, corrected_dir, key_func=lambda stem: stem.split("_")[0])

        assert len(pairs) == 1

    def test_pairs_are_sorted_by_key(self, tmp_path):
        raw_dir, corrected_dir = self._dirs(tmp_path)
        self._touch(raw_dir, ["b_1_1_1_raw.tif", "a_1_1_1_raw.tif"])
        self._touch(corrected_dir, ["b_1_1_1_corrected.tif", "a_1_1_1_corrected.tif"])

        pairs = match_image_pairs(raw_dir, corrected_dir)

        assert [p[0].name[0] for p in pairs] == ["a", "b"]

    def test_no_matches_warns(self, tmp_path):
        raw_dir, corrected_dir = self._dirs(tmp_path)
        self._touch(raw_dir, ["a_1_1_1_raw.tif"])
        self._touch(corrected_dir, ["b_1_1_1_corrected.tif"])

        with pytest.warns(UserWarning, match="No matching image pairs"):
            pairs = match_image_pairs(raw_dir, corrected_dir)

        assert pairs == []

    def test_partial_mismatch_warns_but_returns_matches(self, tmp_path):
        raw_dir, corrected_dir = self._dirs(tmp_path)
        self._touch(raw_dir, ["a_1_1_1_raw.tif", "b_1_1_1_raw.tif"])
        self._touch(corrected_dir, ["a_1_1_1_corrected.tif"])

        with pytest.warns(UserWarning, match="did not have matching pairs"):
            pairs = match_image_pairs(raw_dir, corrected_dir)

        assert len(pairs) == 1

    def test_duplicate_key_within_directory_warns_and_keeps_one(self, tmp_path):
        raw_dir, corrected_dir = self._dirs(tmp_path)
        self._touch(raw_dir, ["a_1_1_1_first.tif", "a_1_1_1_second.tif"])
        self._touch(corrected_dir, ["a_1_1_1_corrected.tif"])

        with pytest.warns(UserWarning, match="Multiple raw files"):
            pairs = match_image_pairs(raw_dir, corrected_dir)

        assert len(pairs) == 1
        assert pairs[0][0].name in {"a_1_1_1_first.tif", "a_1_1_1_second.tif"}

    def test_return_unmatched_includes_leftover_files(self, tmp_path):
        raw_dir, corrected_dir = self._dirs(tmp_path)
        self._touch(raw_dir, ["a_1_1_1_raw.tif", "b_1_1_1_raw.tif"])
        self._touch(corrected_dir, ["a_1_1_1_corrected.tif"])

        pairs, unmatched_raw, unmatched_corrected = match_image_pairs(
            raw_dir, corrected_dir, return_unmatched=True
        )

        assert len(pairs) == 1
        assert [p.name for p in unmatched_raw] == ["b_1_1_1_raw.tif"]
        assert unmatched_corrected == []


class TestComputeRegionSlopes:
    def test_column_gradient_has_positive_col_slope_and_flat_row_slope(self):
        region = np.tile(np.arange(10, dtype=float), (10, 1))  # increases along columns

        _avg_row, _avg_col, row_slope, col_slope, magnitude = _compute_region_slopes(region)

        assert row_slope == pytest.approx(0.0, abs=1e-8)
        assert col_slope == pytest.approx(1.0, abs=1e-8)
        assert magnitude == pytest.approx(1.0, abs=1e-8)

    def test_all_nan_region_returns_nan_slopes(self):
        region = np.full((5, 5), np.nan)

        _avg_row, _avg_col, row_slope, col_slope, _magnitude = _compute_region_slopes(region)

        assert np.isnan(row_slope)
        assert np.isnan(col_slope)


class TestComputeIntensitySlopes:
    def test_shape_mismatch_raises_value_error(self):
        image = np.zeros((10, 10))
        mask = np.zeros((5, 5), dtype=bool)

        with pytest.raises(ValueError, match="does not match mask shape"):
            compute_intensity_slopes(image, mask)

    def test_computes_slopes_for_foreground_and_background_regions(self):
        image = np.tile(np.arange(20, dtype=float), (20, 1))
        mask = np.zeros((20, 20), dtype=bool)
        mask[:, :10] = True  # foreground = left half

        fg_mag, bg_mag, results = compute_intensity_slopes(image, mask)

        assert fg_mag == pytest.approx(1.0, abs=1e-8)
        assert bg_mag == pytest.approx(1.0, abs=1e-8)
        assert set(results) == {
            "fg_avg_row",
            "fg_avg_col",
            "bg_avg_row",
            "bg_avg_col",
            "fg_row_slope",
            "fg_col_slope",
            "bg_row_slope",
            "bg_col_slope",
        }

    def test_uniform_image_has_zero_slope_magnitude(self):
        image = np.full((10, 10), 42.0)
        mask = np.zeros((10, 10), dtype=bool)
        mask[:5, :] = True

        fg_mag, bg_mag, _results = compute_intensity_slopes(image, mask)

        assert fg_mag == pytest.approx(0.0, abs=1e-8)
        assert bg_mag == pytest.approx(0.0, abs=1e-8)


class TestPlotIntensitySlopes:
    def _results(self, size=20):
        image = np.tile(np.arange(size, dtype=float), (size, 1))
        mask = np.zeros((size, size), dtype=bool)
        mask[:, : size // 2] = True
        _fg_mag, _bg_mag, results = compute_intensity_slopes(image, mask)
        return results

    def test_saves_figure_to_path(self, tmp_path):
        save_path = tmp_path / "slopes.png"

        plot_intensity_slopes(self._results(), save_path=save_path)

        assert save_path.exists()

    def test_runs_without_save_path(self):
        plot_intensity_slopes(self._results())  # should not raise; plt.show is mocked
