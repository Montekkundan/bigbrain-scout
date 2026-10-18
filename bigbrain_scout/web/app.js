"use strict";

const $ = (id) => document.getElementById(id);
const axes = ["x", "y", "z"];
let config, checked = null, capsule = null, busy = false;
const numberText = (value) => Number(value.toFixed(7)).toString();
const vectorText = (values) => values.map(numberText).join(" × ");
const byteText = (bytes) => bytes < 1024 ? `${bytes} B` : bytes < 1024 ** 2 ? `${(bytes / 1024).toFixed(1)} KiB` : `${(bytes / 1024 ** 2).toFixed(2)} MiB`;

async function api(path, data) {
  const options = data === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json", "X-Scout-Token": config.csrf_token}, body: JSON.stringify(data)};
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "The workspace refused this operation.");
  return result;
}

function setStatus(text, kind = "") {
  $("status").textContent = text;
  $("status").className = `status ${kind}`;
}
function message(text, error = false) {
  $("message").textContent = text;
  $("message").className = error ? "error" : "";
}
function lock(value) {
  busy = value;
  $("fields").disabled = value;
  $("verify").disabled = value || !capsule;
  $("check").textContent = value ? "Working…" : "Check request";
  $("build").disabled = value || !checked;
}
function enableDownload() {
  $("download").href = `/api/capsules/${capsule.id}/download`;
  $("download").classList.remove("disabled");
  $("download").setAttribute("aria-disabled", "false");
  $("download").tabIndex = 0;
}
function invalidate() {
  checked = null;
  capsule = null;
  $("build").disabled = true;
  $("verify").disabled = true;
  $("download").removeAttribute("href");
  $("download").classList.add("disabled");
  $("download").setAttribute("aria-disabled", "true");
  $("download").tabIndex = -1;
  $("manifest-details").hidden = true;
  $("plan-details").hidden = true;
  $("preview").hidden = true;
  $("preview-placeholder").hidden = false;
  $("preview-note").textContent = "No preview before collection.";
  $("geometry-note").textContent = "Bounds checks are not anatomical validation.";
  $("verification").textContent = "No capsule selected. Existing capsules remain in your local output directory.";
  ["decoded-result", "working-result", "shape-result", "resolution-result"].forEach((id) => $(id).textContent = "Not checked");
  $("actual-shape").textContent = "Not collected";
  $("bounds").replaceChildren();
  const row = document.createElement("tr"), cell = document.createElement("td");
  cell.colSpan = 4; cell.textContent = "Check a request to inspect geometry."; row.append(cell); $("bounds").append(row);
  setStatus("Not checked");
  message("Request changed. Check it again before building a capsule.");
}

function sourceChanged() {
  const source = config.sources.find((item) => item.id === $("source").value);
  const defaults = source.defaults;
  $("source-note").textContent = source.description;
  $("scale").replaceChildren();
  source.scales.forEach((scale) => {
    const option = document.createElement("option"); option.value = scale.key; option.textContent = scale.label; $("scale").append(option);
  });
  $("scale").value = defaults.scale;
  axes.forEach((axis, i) => {$("lo-" + axis).value = defaults.roi_mm[0][i]; $("hi-" + axis).value = defaults.roi_mm[1][i];});
  $("decoded").value = defaults.max_decoded_bytes;
  $("working").value = defaults.max_estimated_working_bytes;
  $("build").textContent = source.id === "synthetic" ? "Build capsule" : "Fetch & build capsule";
  invalidate();
}

function request() {
  return {source_id: $("source").value, roi_mm: ["lo-", "hi-"].map((prefix) => axes.map((axis) => Number($(prefix + axis).value))), scale: $("scale").value, max_decoded_bytes: Number($("decoded").value), max_estimated_working_bytes: Number($("working").value)};
}

function geometry(plan, returned = null) {
  $("decoded-result").textContent = byteText(plan.estimates.decoded_roi_bytes);
  $("working-result").textContent = byteText(plan.estimates.estimated_working_bytes);
  $("shape-result").textContent = plan.shape.join(" × ");
  const actual = returned ? returned.bbox_mm : plan.returned_bbox_mm;
  const label = returned ? "returned" : "planned";
  $("bounds").replaceChildren();
  [["Lower (requested)", plan.requested_bbox_mm[0]], ["Upper (requested)", plan.requested_bbox_mm[1]], [`Lower (${label})`, actual[0]], [`Upper (${label})`, actual[1]]].forEach(([title, values]) => {
    const row = document.createElement("tr");
    [title, ...values.map(numberText)].forEach((value) => {const cell = document.createElement("td"); cell.textContent = value; row.append(cell);});
    $("bounds").append(row);
  });
  $("resolution-result").textContent = `${vectorText(plan.resolution_mm)} (planned)` + (returned ? `\n${vectorText(returned.resolution_mm)} (returned)` : "");
  $("actual-shape").textContent = `${plan.shape.join(" × ")} (planned)` + (returned ? `\n${returned.shape.join(" × ")} (returned)` : "\nNot collected");
  $("geometry-note").textContent = plan.source_id === "synthetic" ? "Synthetic Cartesian grid. Bounds checks are not anatomical validation." : "BigBrain space; voxel-edge bounds. Shapes and resolution use array axis order; see the affine in the manifest. Not anatomical validation.";
  $("plan-json").textContent = JSON.stringify(plan, null, 2);
  $("plan-details").hidden = false;
}

function preview(data, fixtureType, sourceId) {
  const canvas = $("preview");
  const pixels = atob(data.pixels_base64);
  if (pixels.length !== data.width * data.height) throw new Error("Preview byte count is inconsistent.");
  canvas.width = data.width; canvas.height = data.height;
  const context = canvas.getContext("2d"), frame = context.createImageData(data.width, data.height);
  for (let i = 0; i < pixels.length; i++) {
    const value = pixels.charCodeAt(i); frame.data.set([value, value, value, 255], i * 4);
  }
  context.putImageData(frame, 0, 0);
  canvas.hidden = false;
  $("preview-placeholder").hidden = true;
  const label = config.sources.find((item) => item.id === sourceId)?.label || "Selected section";
  $("preview-note").textContent = fixtureType === "synthetic" ? "Synthetic fixture. Not biological data." : `Actual ${label} grayscale bytes. Orientation follows the array, not an anatomical view.`;
}

$("request-form").addEventListener("input", () => {if (!busy) invalidate();});
$("source").addEventListener("change", sourceChanged);
$("request-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  invalidate(); lock(true); setStatus("Checking…");
  message($("source").value === "synthetic" ? "Checking the offline fixture. No network access." : "Reading section metadata, not image bytes. This can take a moment.");
  try {
    checked = await api("/api/plan", request());
    geometry(checked.plan);
    setStatus("Ready to build", "ready");
    message(checked.plan.clipped ? "Request clipped to the source. Inspect planned bounds before building. Estimates are not network bytes or a hard memory cap." : "Request fits the selected source and budgets. No image bytes collected. Estimates are not network bytes or a hard memory cap.");
  } catch (error) {checked = null; setStatus("Request refused", "error"); message(error.message, true);}
  finally {lock(false);}
});

$("build").addEventListener("click", async () => {
  if (busy || !checked) return;
  const token = checked.token; lock(true); setStatus("Collecting…");
  message($("source").value === "synthetic" ? "Building and verifying an offline synthetic capsule." : "Fetching the checked tiny ROI and verifying its returned geometry. Keep live data local until source terms are reviewed.");
  try {
    capsule = await api("/api/build", {token});
    geometry(capsule.plan, capsule.manifest.returned);
    preview(capsule.preview, capsule.manifest.fixture_type, capsule.plan.source_id);
    $("manifest-json").textContent = JSON.stringify(capsule.manifest, null, 2);
    $("manifest-details").hidden = false;
    $("verification").textContent = `Verified geometry and artifact checksums. Capsule ${capsule.id}. ${capsule.verification.warning || ""}`;
    enableDownload();
    setStatus("Capsule built", "ready");
    message(capsule.manifest.fixture_type === "synthetic" ? "Offline capsule built. Download includes a script to reproduce fixture bytes. No live-data claim." : "Live capsule built. Returned geometry passed checks; this is not scientific validation or proof of independent re-download. Keep data local.");
  } catch (error) {capsule = null; setStatus("Collection stopped", "error"); message(error.message, true);}
  finally {checked = null; lock(false);}
});

$("verify").addEventListener("click", async () => {
  if (!capsule || busy) return;
  lock(true);
  try {
    const result = await api("/api/verify", {id: capsule.id});
    $("verification").textContent = `Verified again: geometry and every recorded artifact checksum. ${result.warning || ""}`;
    enableDownload();
  } catch (error) {$("verification").textContent = `Verification refused: ${error.message}`; $("download").removeAttribute("href"); $("download").classList.add("disabled"); $("download").setAttribute("aria-disabled", "true"); $("download").tabIndex = -1;}
  finally {lock(false);}
});

async function start() {
  try {
    config = await api("/api/config");
    $("source").replaceChildren();
    config.sources.forEach((source) => {const option = document.createElement("option"); option.value = source.id; option.textContent = source.label; option.disabled = source.id !== "synthetic" && !config.allow_live; $("source").append(option);});
    sourceChanged(); lock(false);
    message(config.allow_live ? "Start offline or choose a fixed section preset. Live image fetch requires Check request, then Fetch & build capsule." : "Offline mode. Start the CLI with --allow-live to enable fixed section presets. No image bytes collected.");
  } catch (error) {setStatus("Workspace unavailable", "error"); message(`${error.message} Reload after starting the local server.`, true);}
}
start();
