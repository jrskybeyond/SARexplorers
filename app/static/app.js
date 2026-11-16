const form = document.getElementById("analysis-form");
const compareButton = document.getElementById("analyze-button");
const resultMessage = document.getElementById("analysis-result");
const comparison = document.getElementById("raster-comparison");

function formatArea(areaM2) {
  const area = Number(areaM2) || 0;
  return area >= 1_000_000
    ? `${(area / 1_000_000).toFixed(2)} km²`
    : `${Math.round(area).toLocaleString()} m²`;
}

function showMessage(message, isError = false) {
  resultMessage.textContent = message;
  resultMessage.classList.toggle("is-error", isError);
  resultMessage.classList.add("result-message");
}

function clearResult() {
  resultMessage.textContent = "";
  resultMessage.classList.remove("result-message", "is-error");
  comparison.hidden = true;
}

async function compareImages(event) {
  event.preventDefault();
  clearResult();

  const before = form.elements.before.files[0];
  const after = form.elements.after.files[0];
  const files = [before, after];
  const unsupported = files.find((file) => !/\.(tif|tiff)$/i.test(file.name));
  if (unsupported) {
    showMessage(`${unsupported.name} is not a GeoTIFF. Upload .tif or .tiff files.`, true);
    return;
  }

  const oversized = files.find((file) => file.size > 100 * 1024 * 1024);
  if (oversized) {
    showMessage(`${oversized.name} is over the 100 MB limit. Crop it to your study area first.`, true);
    return;
  }

  compareButton.disabled = true;
  compareButton.firstChild.textContent = "Comparing… ";
  showMessage("Comparing the radar images…");

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: new FormData(form),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Image comparison failed.");

    document.getElementById("before-preview").src = result.preview.before;
    document.getElementById("after-preview").src = result.preview.after;
    document.getElementById("difference-preview").src = result.preview.difference.image;
    document.getElementById("before-preview-name").textContent = before.name;
    document.getElementById("after-preview-name").textContent = after.name;

    const candidateArea = Number(result.candidateAreaM2) || 0;
    const threshold = Math.abs(Number(result.thresholdDb));
    showMessage(candidateArea
      ? `Possible radar change: ${formatArea(candidateArea)} showed a return decrease of at least ${threshold} dB. This is not a confirmed flood boundary.`
      : `No area crossed the ${threshold} dB decrease filter. This does not rule out smaller changes or flooding.`);
    comparison.hidden = false;
    comparison.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showMessage(error.message || "Image comparison failed.", true);
  } finally {
    compareButton.disabled = false;
    compareButton.firstChild.textContent = "Compare images ";
  }
}

form.addEventListener("submit", compareImages);
form.addEventListener("change", clearResult);
