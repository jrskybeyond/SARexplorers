from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import os
from typing import Any

import httpx


ASF_SEARCH_URL = os.getenv(
    "ASF_SEARCH_URL",
    "https://api.daac.asf.alaska.edu/services/search/param",
)
CATALOG_TIMEOUT_SECONDS = 40.0
MIN_PAIR_DAYS = 8
MAX_PAIR_DAYS = 16


@dataclass(frozen=True)
class Bounds:
    west: float
    south: float
    east: float
    north: float

    def as_wkt(self) -> str:
        return (
            "POLYGON(("
            f"{self.west} {self.south},"
            f"{self.east} {self.south},"
            f"{self.east} {self.north},"
            f"{self.west} {self.north},"
            f"{self.west} {self.south}"
            "))"
        )


@dataclass(frozen=True)
class Scene:
    scene_id: str
    acquired_at: datetime
    path: int | None
    frame: str | None
    direction: str
    maturity: str
    signature: str
    processing_level: str
    download_url: str
    preview_url: str | None
    geometry: dict[str, Any]

    def to_feature(self) -> dict[str, Any]:
        return {
            "type": "Feature",
            "geometry": self.geometry,
            "properties": {
                "sceneId": self.scene_id,
                "acquiredAt": self.acquired_at.isoformat(),
                "path": self.path,
                "frame": self.frame,
                "direction": self.direction,
                "maturity": self.maturity,
                "processingLevel": self.processing_level,
                "downloadUrl": self.download_url,
                "previewUrl": self.preview_url,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_feature()["properties"] | {"geometry": self.geometry}


class CatalogError(RuntimeError):
    """Raised when ASF returns data that cannot be used by the app."""


def _scene_from_feature(feature: dict[str, Any]) -> Scene | None:
    properties = feature.get("properties") or {}
    scene_id = properties.get("sceneName") or properties.get("fileID")
    geometry = feature.get("geometry")
    acquired = properties.get("startTime")
    download_url = properties.get("url")

    if not (
        isinstance(scene_id, str)
        and isinstance(geometry, dict)
        and isinstance(acquired, str)
        and isinstance(download_url, str)
    ):
        return None

    try:
        acquired_at = datetime.fromisoformat(acquired.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None

    tokens = scene_id.split("_")
    path: int | None = None
    frame: str | None = None
    if len(tokens) > 7:
        try:
            path = int(tokens[5])
        except ValueError:
            path = None
        frame = tokens[7]

    direction_code = tokens[6] if len(tokens) > 6 else ""
    direction = {
        "A": "Ascending",
        "D": "Descending",
    }.get(direction_code, properties.get("flightDirection") or "Unknown")

    if "_PR_" in scene_id or "PROVISIONAL" in download_url:
        maturity = "Provisional"
    elif "_UR_" in scene_id:
        maturity = "ASF UR — verify maturity"
    else:
        maturity = "Unspecified"

    signature = "_".join(tokens[7:11]) if len(tokens) > 10 else ""
    browse_urls = properties.get("browse") or []
    preview_url = next(
        (url for url in browse_urls if "_LATLON.png" in url),
        None,
    )

    return Scene(
        scene_id=scene_id,
        acquired_at=acquired_at,
        path=path,
        frame=frame,
        direction=direction,
        maturity=maturity,
        signature=signature,
        processing_level=properties.get("processingLevel") or "GCOV",
        download_url=download_url,
        preview_url=preview_url,
        geometry=geometry,
    )


def normalize_geojson(
    payload: dict[str, Any],
    maturity_filter: str = "provisional",
) -> list[Scene]:
    if not isinstance(payload, dict):
        raise CatalogError("ASF response was not a GeoJSON object.")
    features = payload.get("features")
    if not isinstance(features, list):
        raise CatalogError("ASF response did not contain a GeoJSON feature list.")

    scenes_by_id: dict[str, Scene] = {}
    for feature in features:
        if not isinstance(feature, dict):
            continue
        scene = _scene_from_feature(feature)
        if scene is None:
            continue
        if maturity_filter == "provisional" and scene.maturity != "Provisional":
            continue
        if maturity_filter == "other" and scene.maturity == "Provisional":
            continue
        scenes_by_id[scene.scene_id] = scene

    return sorted(
        scenes_by_id.values(),
        key=lambda scene: scene.acquired_at,
    )


def find_repeat_pairs(scenes: list[Scene]) -> list[dict[str, Any]]:
    unique_acquisitions: dict[tuple[Any, ...], Scene] = {}
    for scene in scenes:
        acquisition_key = (
            scene.path,
            scene.frame,
            scene.direction,
            scene.maturity,
            scene.signature,
            scene.acquired_at.date(),
        )
        current = unique_acquisitions.get(acquisition_key)
        if current is None or scene.scene_id < current.scene_id:
            unique_acquisitions[acquisition_key] = scene

    acquisitions = sorted(
        unique_acquisitions.values(),
        key=lambda scene: scene.acquired_at,
    )
    pairs: list[dict[str, Any]] = []
    for older_index, older in enumerate(acquisitions):
        for newer in acquisitions[older_index + 1 :]:
            gap_days = (
                newer.acquired_at - older.acquired_at
            ).total_seconds() / 86_400
            if gap_days < MIN_PAIR_DAYS or gap_days > MAX_PAIR_DAYS:
                continue
            if (
                older.path is None
                or older.path != newer.path
                or older.frame != newer.frame
                or older.direction != newer.direction
                or older.maturity != newer.maturity
                or not older.signature
                or older.signature != newer.signature
            ):
                continue
            pairs.append(
                {
                    "before": older.to_dict(),
                    "after": newer.to_dict(),
                    "intervalDays": round(gap_days),
                    "path": older.path,
                    "frame": older.frame,
                    "direction": older.direction,
                    "maturity": older.maturity,
                }
            )

    return sorted(
        pairs,
        key=lambda pair: (
            abs(pair["intervalDays"] - 12),
            -datetime.fromisoformat(pair["after"]["acquiredAt"]).timestamp(),
        ),
    )


def normalize_goff_pairs(
    payload: dict[str, Any],
    *,
    start_date: date,
    end_date: date,
    maturity_filter: str = "provisional",
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise CatalogError("ASF response was not a GeoJSON object.")
    features = payload.get("features")
    if not isinstance(features, list):
        raise CatalogError("ASF response did not contain a GeoJSON feature list.")

    products_by_pair: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for feature in features:
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties") or {}
        product_id = properties.get("sceneName") or properties.get("fileID")
        geometry = feature.get("geometry")
        download_url = properties.get("url")
        if not (
            isinstance(product_id, str)
            and isinstance(geometry, dict)
            and isinstance(download_url, str)
        ):
            continue

        tokens = product_id.split("_")
        if len(tokens) < 20 or tokens[3] != "GOFF":
            continue
        try:
            before_at = datetime.strptime(tokens[11], "%Y%m%dT%H%M%S")
            after_at = datetime.strptime(tokens[13], "%Y%m%dT%H%M%S")
            path = int(tokens[5])
        except (IndexError, ValueError):
            continue
        if (
            before_at.date() < start_date
            or before_at.date() > end_date
            or after_at.date() > end_date
            or after_at <= before_at
        ):
            continue

        url_upper = download_url.upper()
        if "GOFF_VALIDATED" in url_upper:
            maturity = "Validated"
        elif "GOFF_PROVISIONAL" in url_upper:
            maturity = "Provisional"
        elif "GOFF_BETA" in url_upper:
            maturity = "Beta"
        else:
            maturity = "Unspecified"
        if maturity_filter == "provisional" and maturity != "Provisional":
            continue

        browse_urls = properties.get("browse") or []
        preview_url = next(
            (
                url
                for url in browse_urls
                if isinstance(url, str)
                and "_LATLON.png" in url
                and "thumbnail" not in url.lower()
            ),
            None,
        )
        processing_date = properties.get("processingDate")
        product = {
            "productId": product_id,
            "beforeAt": before_at.isoformat() + "Z",
            "afterAt": after_at.isoformat() + "Z",
            "beforeDate": before_at.date().isoformat(),
            "afterDate": after_at.date().isoformat(),
            "intervalDays": (after_at.date() - before_at.date()).days,
            "path": path,
            "frame": tokens[7],
            "direction": {
                "A": "Ascending",
                "D": "Descending",
            }.get(tokens[6], "Unknown"),
            "maturity": maturity,
            "crid": tokens[15],
            "processingDate": processing_date if isinstance(processing_date, str) else "",
            "downloadUrl": download_url,
            "previewUrl": preview_url,
            "geometry": geometry,
        }
        pair_key = (
            before_at,
            after_at,
            path,
            tokens[7],
            tokens[6],
            tokens[9],
            tokens[10],
        )
        products_by_pair.setdefault(pair_key, []).append(product)

    maturity_rank = {"Validated": 3, "Provisional": 2, "Beta": 1, "Unspecified": 0}
    pairs = []
    for products in products_by_pair.values():
        products.sort(
            key=lambda product: (
                maturity_rank[product["maturity"]],
                product["processingDate"],
                product["crid"],
            ),
            reverse=True,
        )
        selected = products[0]
        selected["variantCount"] = len(products)
        selected["alternateProducts"] = [
            {
                "productId": product["productId"],
                "crid": product["crid"],
                "maturity": product["maturity"],
                "processingDate": product["processingDate"],
                "downloadUrl": product["downloadUrl"],
            }
            for product in products[1:]
        ]
        pairs.append(selected)

    return sorted(pairs, key=lambda pair: pair["beforeAt"])


async def search_nisar_gc_scene_catalog(
    client: httpx.AsyncClient,
    *,
    start_date: date,
    end_date: date,
    bounds: Bounds,
    maturity_filter: str = "provisional",
) -> dict[str, Any]:
    params = {
        "dataset": "NISAR",
        "processingLevel": "GCOV",
        "intersectsWith": bounds.as_wkt(),
        "start": f"{start_date.isoformat()}T00:00:00Z",
        "end": f"{end_date.isoformat()}T23:59:59Z",
        "output": "geojson",
    }

    try:
        response = await client.get(ASF_SEARCH_URL, params=params)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise CatalogError(f"Unable to query the ASF NISAR catalog: {error}") from error

    scenes = normalize_geojson(payload, maturity_filter=maturity_filter)
    return {
        "type": "FeatureCollection",
        "features": [scene.to_feature() for scene in scenes],
        "scenes": [scene.to_dict() for scene in scenes],
        "pairs": find_repeat_pairs(scenes),
        "count": len(scenes),
        "source": "NASA Alaska Satellite Facility (ASF) Search API",
    }


async def search_nisar_goff_pairs(
    client: httpx.AsyncClient,
    *,
    start_date: date,
    end_date: date,
    bounds: Bounds,
    maturity_filter: str = "provisional",
) -> dict[str, Any]:
    params = {
        "dataset": "NISAR",
        "processingLevel": "GOFF",
        "intersectsWith": bounds.as_wkt(),
        "start": f"{start_date.isoformat()}T00:00:00Z",
        "end": f"{end_date.isoformat()}T23:59:59Z",
        "output": "geojson",
    }

    try:
        response = await client.get(ASF_SEARCH_URL, params=params)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise CatalogError(f"Unable to query the ASF NISAR GOFF catalog: {error}") from error

    pairs = normalize_goff_pairs(
        payload,
        start_date=start_date,
        end_date=end_date,
        maturity_filter=maturity_filter,
    )
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": pair["geometry"],
                "properties": {
                    key: value for key, value in pair.items() if key != "geometry"
                },
            }
            for pair in pairs
        ],
        "pairs": pairs,
        "count": len(pairs),
        "source": "NASA Alaska Satellite Facility (ASF) Search API",
    }
