from __future__ import annotations

import base64
import struct
from typing import Any, Literal
import zlib

import numpy as np
import rasterio
from rasterio.errors import RasterioError
from rasterio.features import shapes
from rasterio.warp import transform_geom


MAX_RASTER_PIXELS = 10_000_000
MAX_OUTPUT_FEATURES = 300
PREVIEW_MAX_DIMENSION = 1200
PREVIEW_SAMPLE_TARGET = 500_000
CHANGE_OVERLAY_DISPLAY_THRESHOLD_DB = 1.0


class AnalysisInputError(ValueError):
    """Raised when uploaded rasters are not suitable for a comparison."""


def _ring_area(ring: list[list[float]]) -> float:
    if len(ring) < 3:
        return 0.0
    return abs(
        sum(
            (ring[index][0] * ring[(index + 1) % len(ring)][1])
            - (ring[(index + 1) % len(ring)][0] * ring[index][1])
            for index in range(len(ring))
        )
        / 2.0
    )


def _geometry_area(geometry: dict[str, Any]) -> float:
    coordinates = geometry.get("coordinates", [])
    if geometry.get("type") == "Polygon":
        if not coordinates:
            return 0.0
        return max(
            0.0,
            _ring_area(coordinates[0])
            - sum(_ring_area(ring) for ring in coordinates[1:]),
        )
    if geometry.get("type") == "MultiPolygon":
        return sum(
            max(
                0.0,
                _ring_area(polygon[0])
                - sum(_ring_area(ring) for ring in polygon[1:]),
            )
            for polygon in coordinates
            if polygon
        )
    return 0.0


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def _rgba_png_data_url(rgba: np.ndarray) -> str:
    scanlines = b"".join(
        b"\x00" + rgba[row].tobytes()
        for row in range(rgba.shape[0])
    )
    height, width = rgba.shape[:2]
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(scanlines))
        + _png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _preview_data_url(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    scale: Literal["linear", "db"],
    display_min: float,
    display_max: float,
) -> str:
    step = max(1, int(np.ceil(max(values.shape) / PREVIEW_MAX_DIMENSION)))
    sampled = values[::step, ::step]
    sampled_valid = valid[::step, ::step]
    display_values = sampled[sampled_valid]
    if scale == "linear":
        display_values = 10.0 * np.log10(display_values)

    pixels = np.zeros(sampled.shape, dtype=np.uint8)
    pixels[sampled_valid] = np.rint(
        np.clip(
            (display_values - display_min) / (display_max - display_min),
            0.0,
            1.0,
        )
        * 255
    ).astype(np.uint8)
    rgba = np.empty((*sampled.shape, 2), dtype=np.uint8)
    rgba[:, :, 0] = pixels
    rgba[:, :, 1] = sampled_valid.astype(np.uint8) * 255
    scanlines = b"".join(
        b"\x00" + rgba[row].tobytes()
        for row in range(rgba.shape[0])
    )
    height, width = sampled.shape
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 4, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(scanlines))
        + _png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _box_mean_preview(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    window_size: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    radius = window_size // 2
    height, width = values.shape

    def summed_area(source: np.ndarray) -> np.ndarray:
        return np.pad(source, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)

    weighted = np.where(valid, values, 0).astype("float64")
    support = valid.astype("float64")
    value_integral = summed_area(weighted)
    support_integral = summed_area(support)
    rows = np.arange(height)
    columns = np.arange(width)
    top = np.maximum(rows - radius, 0)
    bottom = np.minimum(rows + radius + 1, height)
    left = np.maximum(columns - radius, 0)
    right = np.minimum(columns + radius + 1, width)

    def window_sum(integral: np.ndarray) -> np.ndarray:
        return (
            integral[bottom[:, np.newaxis], right[np.newaxis, :]]
            - integral[top[:, np.newaxis], right[np.newaxis, :]]
            - integral[bottom[:, np.newaxis], left[np.newaxis, :]]
            + integral[top[:, np.newaxis], left[np.newaxis, :]]
        )

    value_sum = window_sum(value_integral)
    valid_count = window_sum(support_integral)
    smoothed_valid = valid & (valid_count >= (window_size * window_size + 1) // 2)
    smoothed = np.zeros(values.shape, dtype=np.float32)
    smoothed[smoothed_valid] = (
        value_sum[smoothed_valid] / valid_count[smoothed_valid]
    ).astype(np.float32)
    return smoothed, smoothed_valid


def _difference_preview_data_url(
    delta_db: np.ndarray,
    valid: np.ndarray,
) -> tuple[str, float]:
    step = max(1, int(np.ceil(max(delta_db.shape) / PREVIEW_MAX_DIMENSION)))
    sampled = delta_db[::step, ::step]
    sampled_valid = valid[::step, ::step]
    sampled, sampled_valid = _box_mean_preview(sampled, sampled_valid)
    absolute_values = np.abs(sampled[sampled_valid])
    changed_values = absolute_values[absolute_values > 0]
    display_limit = (
        float(np.percentile(changed_values, 98))
        if changed_values.size
        else 0.0
    )
    color_limit = display_limit if display_limit > 0 else 1.0
    normalized = np.clip(sampled / color_limit, -1.0, 1.0)
    magnitude = np.abs(normalized)[:, :, np.newaxis]
    neutral = np.array([245, 245, 240], dtype=np.float32)
    endpoints = np.where(
        (normalized < 0)[:, :, np.newaxis],
        np.array([205, 64, 55], dtype=np.float32),
        np.array([44, 115, 177], dtype=np.float32),
    )
    rgb = np.rint(neutral + magnitude * (endpoints - neutral)).astype(np.uint8)
    rgba = np.empty((*sampled.shape, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = sampled_valid.astype(np.uint8) * 255
    return _rgba_png_data_url(rgba), display_limit


def _change_overlay_data_url(
    delta_db: np.ndarray,
    valid: np.ndarray,
) -> tuple[str, int]:
    step = max(1, int(np.ceil(max(delta_db.shape) / PREVIEW_MAX_DIMENSION)))
    sampled = delta_db[::step, ::step]
    sampled_valid = valid[::step, ::step]
    sampled, sampled_valid = _box_mean_preview(sampled, sampled_valid)
    highlighted = sampled_valid & (sampled != 0) & (
        np.abs(sampled) >= CHANGE_OVERLAY_DISPLAY_THRESHOLD_DB
    )

    rgba = np.zeros((*sampled.shape, 4), dtype=np.uint8)
    lower = highlighted & (sampled < 0)
    higher = highlighted & (sampled > 0)
    rgba[lower] = [230, 61, 55, 205]
    rgba[higher] = [37, 132, 220, 205]
    return _rgba_png_data_url(rgba), int(highlighted.sum())


def _raster_previews(
    before: np.ndarray,
    after: np.ndarray,
    valid: np.ndarray,
    *,
    scale: Literal["linear", "db"],
    delta_db: np.ndarray,
) -> dict[str, Any]:
    step = max(
        1,
        int(np.ceil(np.sqrt(before.size / PREVIEW_SAMPLE_TARGET))),
    )
    before_sample = before[::step, ::step]
    after_sample = after[::step, ::step]
    valid_sample = valid[::step, ::step]
    before_values = before_sample[valid_sample]
    after_values = after_sample[valid_sample]
    if scale == "linear":
        before_values = 10.0 * np.log10(before_values)
        after_values = 10.0 * np.log10(after_values)
    sample_values = np.concatenate((before_values, after_values))

    if sample_values.size:
        display_min, display_max = np.percentile(sample_values, [2, 98])
        display_min = float(display_min)
        display_max = float(display_max)
        if display_max <= display_min:
            display_min -= 0.5
            display_max += 0.5
        display_range = {
            "minDb": round(display_min, 2),
            "maxDb": round(display_max, 2),
        }
    else:
        display_min, display_max = 0.0, 1.0
        display_range = None

    difference_image, difference_limit = _difference_preview_data_url(
        delta_db,
        valid,
    )
    overlay_image, overlay_pixel_count = _change_overlay_data_url(
        delta_db,
        valid,
    )
    return {
        "before": _preview_data_url(
            before,
            valid,
            scale=scale,
            display_min=display_min,
            display_max=display_max,
        ),
        "after": _preview_data_url(
            after,
            valid,
            scale=scale,
            display_min=display_min,
            display_max=display_max,
        ),
        "displayRangeDb": display_range,
        "validPixelCount": int(valid.sum()),
        "difference": {
            "image": difference_image,
            "displayLimitDb": round(difference_limit, 3),
            "displayFilter": "5x5 preview-pixel mean",
            "changedPixelCount": int(np.count_nonzero(delta_db[valid])),
        },
        "changeOverlay": {
            "image": overlay_image,
            "thresholdDb": CHANGE_OVERLAY_DISPLAY_THRESHOLD_DB,
            "displayFilter": "5x5 preview-pixel mean",
            "highlightedPixelCount": overlay_pixel_count,
        },
    }


def analyze_backscatter_change(
    before_path: str,
    after_path: str,
    *,
    scale: Literal["linear", "db"],
    threshold_db: float,
    minimum_area_m2: float = 1_000.0,
) -> dict[str, Any]:
    if not -25.0 <= threshold_db <= 0.0:
        raise AnalysisInputError("Threshold must be between -25 and 0 dB.")
    if scale not in ("linear", "db"):
        raise AnalysisInputError("Input scale must be linear or db.")

    try:
        with rasterio.open(before_path) as before_ds, rasterio.open(after_path) as after_ds:
            if before_ds.count != 1 or after_ds.count != 1:
                raise AnalysisInputError("Each GeoTIFF must contain exactly one band.")
            if before_ds.width != after_ds.width or before_ds.height != after_ds.height:
                raise AnalysisInputError(
                    "The two rasters must have matching dimensions. Align/crop them first."
                )
            if before_ds.crs is None or after_ds.crs is None or before_ds.crs != after_ds.crs:
                raise AnalysisInputError(
                    "Both rasters need the same defined coordinate reference system."
                )
            if before_ds.transform != after_ds.transform:
                raise AnalysisInputError(
                    "The rasters must have matching pixel grids. Align them first."
                )
            if before_ds.width * before_ds.height > MAX_RASTER_PIXELS:
                raise AnalysisInputError(
                    f"Raster is too large; maximum supported size is "
                    f"{MAX_RASTER_PIXELS:,} pixels."
                )
            if before_ds.crs.linear_units not in ("metre", "meter"):
                raise AnalysisInputError(
                    "Use a projected raster in metres so the app can estimate area."
                )

            before = before_ds.read(1, masked=True).astype("float32")
            after = after_ds.read(1, masked=True).astype("float32")
            transform = before_ds.transform
            source_crs = before_ds.crs
            nodata_mask = np.ma.getmaskarray(before) | np.ma.getmaskarray(after)
            before_values = np.asarray(before.filled(np.nan), dtype=np.float32)
            after_values = np.asarray(after.filled(np.nan), dtype=np.float32)
    except RasterioError as error:
        raise AnalysisInputError(
            "Could not read one of the files as a supported GeoTIFF."
        ) from error

    valid = (
        ~nodata_mask
        & np.isfinite(before_values)
        & np.isfinite(after_values)
    )
    if scale == "linear":
        valid &= (before_values > 0) & (after_values > 0)
        delta_db = np.zeros(before_values.shape, dtype=np.float32)
        delta_db[valid] = 10.0 * np.log10(
            after_values[valid] / before_values[valid]
        )
    else:
        delta_db = after_values - before_values

    candidate_mask = valid & (delta_db <= threshold_db)
    preview = _raster_previews(
        before_values,
        after_values,
        valid,
        scale=scale,
        delta_db=delta_db,
    )
    pixel_area_m2 = abs(
        transform.a * transform.e - transform.b * transform.d
    )
    total_candidate_area_m2 = float(candidate_mask.sum() * pixel_area_m2)
    if not candidate_mask.any():
        return {
            "type": "FeatureCollection",
            "features": [],
            "candidateAreaM2": 0.0,
            "meanChangeDb": None,
            "thresholdDb": threshold_db,
            "method": "Backscatter decrease candidate; not a confirmed flood map.",
            "truncated": False,
            "preview": preview,
        }

    features: list[dict[str, Any]] = []
    mapped_area_m2 = 0.0
    polygon_count = 0
    truncated = False
    for raw_geometry, value in shapes(
        candidate_mask.astype("uint8"),
        mask=candidate_mask,
        transform=transform,
        connectivity=8,
    ):
        if int(value) != 1:
            continue
        polygon_count += 1
        area_m2 = _geometry_area(raw_geometry)
        if area_m2 < minimum_area_m2:
            continue
        mapped_area_m2 += area_m2
        if len(features) >= MAX_OUTPUT_FEATURES:
            truncated = True
            continue
        geometry = transform_geom(
            source_crs,
            "EPSG:4326",
            raw_geometry,
            precision=6,
        )
        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {"areaM2": round(area_m2, 1)},
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
        "candidateAreaM2": round(total_candidate_area_m2, 1),
        "mappedAreaM2": round(mapped_area_m2, 1),
        "meanChangeDb": round(float(delta_db[candidate_mask].mean()), 2),
        "polygonCount": polygon_count,
        "thresholdDb": threshold_db,
        "method": "Backscatter decrease candidate; not a confirmed flood map.",
        "truncated": truncated,
        "preview": preview,
    }
