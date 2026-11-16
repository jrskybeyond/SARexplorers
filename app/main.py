from __future__ import annotations

from datetime import date
import logging
import os
from pathlib import Path
import tempfile
from typing import Literal

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from app.services.asf_catalog import (
    Bounds,
    CatalogError,
    search_nisar_gc_scene_catalog,
    search_nisar_goff_pairs,
)
from app.services.change_detection import (
    AnalysisInputError,
    analyze_backscatter_change,
)


ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_DATE_RANGE_DAYS = 180
MAX_GLACIER_DATE_RANGE_DAYS = 400
RASUWA_FLOOD_DATE = date(2026, 8, 26)
GOFF_EVENT_WINDOW_START = date(2026, 8, 15)
GOFF_EVENT_WINDOW_END = date(2026, 9, 12)
logger = logging.getLogger("rasuwa_floodbeat")

app = FastAPI(
    title="SAR Explorer",
    description=(
        "NISAR Change Viewer: compare before-and-after GCOV GeoTIFFs for radar-return changes "
        "around Rasuwa, Nepal."
    ),
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "rasuwa-floodbeat"}


@app.get("/api/catalog")
async def catalog(
    start: date = Query(default=date(2026, 8, 19)),
    end: date = Query(default=date(2026, 8, 31)),
    west: float = Query(default=85.0, ge=-180, le=180),
    south: float = Query(default=27.8, ge=-90, le=90),
    east: float = Query(default=86.0, ge=-180, le=180),
    north: float = Query(default=28.5, ge=-90, le=90),
    maturity: Literal["provisional", "other", "all"] = "provisional",
) -> dict:
    if start > end:
        raise HTTPException(
            status_code=422,
            detail="Start date must be on or before end date.",
        )
    if (end - start).days > MAX_DATE_RANGE_DAYS:
        raise HTTPException(
            status_code=422,
            detail=f"Date range cannot exceed {MAX_DATE_RANGE_DAYS} days.",
        )
    if west >= east or south >= north:
        raise HTTPException(
            status_code=422,
            detail="The AOI bounds must have west < east and south < north.",
        )

    bounds = Bounds(west=west, south=south, east=east, north=north)
    try:
        async with httpx.AsyncClient(timeout=40.0) as client:
            result = await search_nisar_gc_scene_catalog(
                client,
                start_date=start,
                end_date=end,
                bounds=bounds,
                maturity_filter=maturity,
            )
    except CatalogError as error:
        logger.warning("ASF catalog query failed: %s", error)
        raise HTTPException(status_code=502, detail=str(error)) from error

    result["query"] = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "bounds": {
            "west": west,
            "south": south,
            "east": east,
            "north": north,
        },
        "maturity": maturity,
    }
    return result


@app.get("/api/glacier-pairs")
async def glacier_pairs(
    start: date = Query(default=date(2025, 11, 28)),
    end: date = Query(default=date(2026, 8, 25)),
    west: float = Query(default=85.45, ge=-180, le=180),
    south: float = Query(default=28.30, ge=-90, le=90),
    east: float = Query(default=85.75, ge=-180, le=180),
    north: float = Query(default=28.50, ge=-90, le=90),
    maturity: Literal["provisional", "all"] = "provisional",
) -> dict:
    if start > end:
        raise HTTPException(
            status_code=422,
            detail="Start date must be on or before end date.",
        )
    if (end - start).days > MAX_GLACIER_DATE_RANGE_DAYS:
        raise HTTPException(
            status_code=422,
            detail=f"Date range cannot exceed {MAX_GLACIER_DATE_RANGE_DAYS} days.",
        )
    if end >= RASUWA_FLOOD_DATE:
        raise HTTPException(
            status_code=422,
            detail="Pre-event GOFF search must end before August 26, 2026.",
        )
    if west >= east or south >= north:
        raise HTTPException(
            status_code=422,
            detail="The AOI bounds must have west < east and south < north.",
        )

    bounds = Bounds(west=west, south=south, east=east, north=north)
    try:
        async with httpx.AsyncClient(timeout=40.0) as client:
            result = await search_nisar_goff_pairs(
                client,
                start_date=start,
                end_date=end,
                bounds=bounds,
                maturity_filter=maturity,
            )
    except CatalogError as error:
        logger.warning("ASF GOFF catalog query failed: %s", error)
        raise HTTPException(status_code=502, detail=str(error)) from error

    result["query"] = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "bounds": {
            "west": west,
            "south": south,
            "east": east,
            "north": north,
        },
        "maturity": maturity,
        "eventDate": RASUWA_FLOOD_DATE.isoformat(),
    }
    return result


@app.get("/api/glacier-event-pairs")
async def glacier_event_pairs() -> dict:
    bounds = Bounds(west=85.45, south=28.30, east=85.75, north=28.50)
    try:
        async with httpx.AsyncClient(timeout=40.0) as client:
            result = await search_nisar_goff_pairs(
                client,
                start_date=GOFF_EVENT_WINDOW_START,
                end_date=GOFF_EVENT_WINDOW_END,
                bounds=bounds,
                maturity_filter="all",
            )
    except CatalogError as error:
        logger.warning("ASF GOFF event-window query failed: %s", error)
        raise HTTPException(status_code=502, detail=str(error)) from error

    crossing_pairs = []
    for pair in result["pairs"]:
        before_date = date.fromisoformat(pair["beforeDate"])
        after_date = date.fromisoformat(pair["afterDate"])
        if before_date < RASUWA_FLOOD_DATE <= after_date:
            pair["daysBeforeEvent"] = (RASUWA_FLOOD_DATE - before_date).days
            pair["daysAfterEvent"] = (after_date - RASUWA_FLOOD_DATE).days
            crossing_pairs.append(pair)

    result["pairs"] = crossing_pairs
    result["features"] = [
        {
            "type": "Feature",
            "geometry": pair["geometry"],
            "properties": {
                key: value for key, value in pair.items() if key != "geometry"
            },
        }
        for pair in crossing_pairs
    ]
    result["count"] = len(crossing_pairs)
    result["query"] = {
        "windowStart": GOFF_EVENT_WINDOW_START.isoformat(),
        "windowEnd": GOFF_EVENT_WINDOW_END.isoformat(),
        "eventDate": RASUWA_FLOOD_DATE.isoformat(),
        "bounds": {
            "west": bounds.west,
            "south": bounds.south,
            "east": bounds.east,
            "north": bounds.north,
        },
        "maturity": "all",
    }
    return result


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    name = (upload.filename or "").lower()
    if not name.endswith((".tif", ".tiff")):
        raise HTTPException(
            status_code=415,
            detail=f"{upload.filename or 'Uploaded file'} must be a GeoTIFF.",
        )

    size = 0
    with destination.open("wb") as target:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Each GeoTIFF must be 100 MB or smaller.",
                )
            target.write(chunk)
    await upload.close()
    if size == 0:
        raise HTTPException(status_code=400, detail="Uploaded GeoTIFF is empty.")


@app.post("/api/analyze")
async def analyze(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    scale: Literal["linear", "db"] = Form(default="linear"),
    threshold_db: float = Form(default=-3.0, ge=-25.0, le=0.0),
) -> dict:
    with tempfile.TemporaryDirectory(prefix="rasuwa-floodbeat-") as temp_dir:
        before_path = Path(temp_dir) / "before.tif"
        after_path = Path(temp_dir) / "after.tif"
        await _save_upload(before, before_path)
        await _save_upload(after, after_path)
        try:
            return await run_in_threadpool(
                analyze_backscatter_change,
                str(before_path),
                str(after_path),
                scale=scale,
                threshold_db=threshold_db,
            )
        except AnalysisInputError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=True,
    )
