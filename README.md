# SAR Explorer

## NISAR Change Viewer · Rasuwa

A simple before-and-after viewer for NASA-ISRO NISAR radar data over Rasuwa,
Nepal. Upload two matching GeoTIFFs to view the source images and a color
preview of their radar-return difference.

## What it does

- Shows the uploaded before and after images side by side.
- Creates a radar-return difference preview and highlights possible decreases.
- Reports candidate changed area as a screening result, not a confirmed flood
  boundary or damage map.

## What you need

The app currently accepts single-band `.tif` or `.tiff` files (up to 100 MB
each). The two rasters must have the same dimensions, projection, pixel grid,
and units in metres. Export the same GCOV layer and polarization for both
dates, then crop and align them before upload.

This version is upload-only: it does not sign in to Earthdata, download NISAR
HDF5 products, or convert HDF5 to GeoTIFF. NISAR downloads require an
Earthdata Login. See [start-app.md](./start-app.md) for the short ASF and
GeoTIFF preparation workflow.

## Run locally on Windows

Use Python 3.11 or newer:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --reload-dir app
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Interpretation

The difference image shows radar-return changes between the uploaded dates.
Lower or higher return can have several causes; neither is proof of
inundation. Use candidate areas to prioritize checks with local reports or
field teams. This prototype is not an official flood map or warning system.

## Project submission materials

- [Seven-slide deck](docs/SAR-Explorer-7-slide-deck.pptx)
- [Slide outline](docs/competition-deck.md)
- [Submission content and references](docs/project-submission-content.md)

## Tests

```powershell
pytest
```
