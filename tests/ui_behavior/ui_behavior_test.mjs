// Behavioral UI test: real app.js (jsdom DOM) against the REAL app handlers
// served by server.py (fake two-GPU inventory, no actual GPU needed).
//
// Checks: device selector population, Save defaults round-trip through the
// settings API, selected-device VRAM display, Test button wiring, and
// per-job device overrides appended to the submission FormData.

import { JSDOM } from "jsdom";
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.dirname(path.dirname(HERE)); // tests/ui_behavior -> repo root
const PORT = process.env.UI_TEST_PORT || "8791";
const BASE = `http://127.0.0.1:${PORT}`;
const GPU_3060 = "GPU-2aeb82bb-6642-0bb1-4304-c24ebbda745c";
const GPU_1050 = "GPU-60ce4d58-ba50-b3fa-82da-3e51b259e857";

const html = fs.readFileSync(`${REPO}/templates/index.html`, "utf8");
const appJs = fs.readFileSync(`${REPO}/static/js/app.js`, "utf8");

const failures = [];
function check(name, cond, detail = "") {
  if (cond) {
    console.log(`  PASS ${name}`);
  } else {
    failures.push(name + (detail ? ` (${detail})` : ""));
    console.log(`  FAIL ${name} ${detail}`);
  }
}

function wait(condFn, timeoutMs = 15000, label = "condition") {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      let v;
      try { v = condFn(); } catch (e) { v = false; }
      if (v) return resolve(v);
      if (Date.now() - started > timeoutMs) {
        return reject(new Error(`timeout waiting for ${label}`));
      }
      setTimeout(tick, 50);
    };
    tick();
  });
}

// Minimal real-HTTP fetch polyfill bound to the test server.
function realFetch(url, opts = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request(
      BASE + url,
      {
        method: opts.method || "GET",
        headers: opts.headers || {},
      },
      (res) => {
        let body = "";
        res.on("data", (c) => (body += c));
        res.on("end", () =>
          resolve({
            ok: res.statusCode >= 200 && res.statusCode < 300,
            status: res.statusCode,
            text: async () => body,
            json: async () => JSON.parse(body),
          })
        );
      }
    );
    req.on("error", reject);
    if (opts.body) req.write(opts.body);
    req.end();
  });
}

const dom = new JSDOM(html, {
  url: BASE + "/",
  runScripts: "outside-only",
  pretendToBeVisual: true,
});
const { window } = dom;

window.fetch = (url, opts) => realFetch(url, opts);

// Minimal XHR polyfill: records the FormData, then answers 200.
const xhrCalls = [];
class FakeXHR {
  constructor() {
    this.upload = { addEventListener() {} };
    this._listeners = {};
  }
  open(method, url) {
    this._method = method;
    this._url = url;
  }
  addEventListener(ev, fn) {
    this._listeners[ev] = fn;
  }
  send(formData) {
    xhrCalls.push({ method: this._method, url: this._url, formData });
    setTimeout(() => {
      this.status = 200;
      this.responseText = '{"job_id": "ui-test-job"}';
      if (this._listeners.load) this._listeners.load();
    }, 10);
  }
}
window.XMLHttpRequest = FakeXHR;

// jsdom has no EventSource; stub it so the post-upload SSE path can run.
window.EventSource = class {
  constructor() { this.readyState = 0; }
  close() { this.readyState = 2; }
  addEventListener() {}
  onopen() {}
  onmessage() {}
  onerror() {}
};

// Silence jsdom's "not implemented" console spam.
window._virtualConsole = dom.virtualConsole;
dom.virtualConsole.on("jsdomError", () => {});

// Run the real app.js against the parsed DOM.
window.eval(appJs);

const $ = (id) => window.document.getElementById(id);

async function main() {
  // --- 1. Device selectors populate from the real /api/compute/devices ---
  const trans = $("computeTranscriptionDevice");
  const diar = $("computeDiarizationDevice");
  await wait(
    () => trans && trans.options.length >= 4,
    15000,
    "device selector population"
  );
  const values = [...trans.options].map((o) => o.value);
  check("transcription select offers 'Not selected'", values.includes(""));
  check("transcription select offers CPU", values.includes("cpu"));
  check("transcription select lists both GPUs", values.includes(GPU_3060) && values.includes(GPU_1050));
  const gpuOption = [...trans.options].find((o) => o.value === GPU_1050);
  check("GPU option label shows name + free VRAM",
    /1050 Ti/.test(gpuOption.textContent) && /free/.test(gpuOption.textContent),
    gpuOption.textContent);
  const diarValues = [...diar.options].map((o) => o.value);
  check("diarization select offers 'same'", diarValues.includes("same"));
  const advTrans = $("advTranscriptionDevice");
  check("advanced transcription override defaults to saved default (empty)",
    advTrans && advTrans.value === "", `value='${advTrans && advTrans.value}'`);

  // --- 2. Save defaults round-trips through the real settings API ---
  trans.value = GPU_1050;
  diar.value = "same";
  $("computeWhisperPrecision").value = "auto";
  $("computeSaveBtn").click();
  await wait(
    () => $("computeStatus") && $("computeStatus").textContent.startsWith("Saved."),
    10000,
    "save status"
  );
  const settings = await (await realFetch("/api/compute/settings")).json();
  check("saved default persisted to transcription role",
    settings.resolved && settings.resolved.id === GPU_1050,
    JSON.stringify(settings.resolved));
  check("saved diarization follows transcription",
    settings.settings.diarization.device.backend === "same_as_transcription");
  check("saved precision persisted",
    settings.settings.transcription.compute_type === "auto");
  const reloaded = await (await realFetch("/api/compute/settings")).json();
  check("resolved device name resolved", reloaded.resolved_device_name && /1050 Ti/.test(reloaded.resolved_device_name), reloaded.resolved_device_name);

  // --- 3. VRAM monitor shows the SELECTED device ---
  await wait(
    () => {
      const label = $("vramLabel");
      return label && /1050 Ti/.test(label.textContent);
    },
    15000,
    "selected-device VRAM label"
  );
  check("VRAM label names the selected device", true);
  const label = $("vramLabel");
  check("VRAM label shows used/total of that device",
    /\d/.test(label.textContent) && /\//.test(label.textContent), label.textContent);

  // --- 4. Test button wiring (probe will fail in sandbox: no worker interpreter) ---
  $("computeTestBtn").click();
  await wait(
    () => {
      const el = $("computeTestResult");
      return el && el.textContent && !el.textContent.startsWith("Running");
    },
    60000,
    "compute test result"
  );
  const testResult = $("computeTestResult").textContent;
  check("Test button produced a rendered result", testResult.length > 0, testResult.slice(0, 120));
  console.log("  (test result content: " + testResult.replace(/\n/g, " | ").slice(0, 200) + ")");

  // --- 5. Per-job override flows into the submission FormData ---
  const file = new window.File([new Uint8Array(64)], "interview.mp3", { type: "audio/mpeg" });
  Object.defineProperty($("fileInput"), "files", { value: [file], configurable: true });
  $("fileInput").dispatchEvent(new window.Event("change", { bubbles: true }));
  await wait(() => $("fileName") && $("fileName").textContent.length > 0, 5000, "file selected");
  check("file selection reflected in UI", /interview\.mp3/.test($("fileName").textContent));

  advTrans.value = GPU_3060; // per-job override differs from the saved default
  $("btnTranscribe").click();
  await wait(() => xhrCalls.length > 0, 10000, "submission XHR");
  const submission = xhrCalls.find((c) => c.url === "/api/transcribe");
  check("submission FormData built for /api/transcribe", Boolean(submission));
  const sentDevice = submission && submission.formData.get("transcription_device");
  check("per-job transcription override included in submission", sentDevice === GPU_3060, String(sentDevice));
  check("diarization override empty when unset (saved default applies)",
    submission && !submission.formData.get("diarization_device"));

  // Wait for the (stubbed) response to be processed before lifecycle checks:
  // the success path consumes the file, hides the info bar, and leaves the
  // button disabled until a new file is picked.
  await wait(
    () => $("fileInfo").hidden === true && $("uploadProgressText").textContent.includes("Upload complete"),
    10000,
    "upload success handling"
  );
  check("successful upload consumes the file and disables transcribe",
    $("btnTranscribe").disabled);
  const file2 = new window.File([new Uint8Array(64)], "interview2.mp3", { type: "audio/mpeg" });
  Object.defineProperty($("fileInput"), "files", { value: [file2], configurable: true });
  $("fileInput").dispatchEvent(new window.Event("change", { bubbles: true }));
  await wait(() => !$("btnTranscribe").disabled, 5000, "button re-enabled on new file");
  check("selecting a new file re-enables the button", true);

  // --- 6. A CPU selection is valid: VRAM panel says so, not 'no valid device' ---
  trans.value = "cpu";
  $("computeSaveBtn").click();
  await wait(
    () => $("computeStatus") && $("computeStatus").textContent.startsWith("Saved."),
    10000,
    "cpu save status"
  );
  await wait(
    () => {
      const label = $("vramLabel");
      return label && /CPU selected/.test(label.textContent);
    },
    15000,
    "cpu VRAM label"
  );
  check("CPU selection shows 'CPU selected — no GPU VRAM'",
    /CPU selected/.test($("vramLabel").textContent), $("vramLabel").textContent);

  // --- report ---
  if (failures.length) {
    console.error(`\n${failures.length} FAILURE(S):`);
    failures.forEach((f) => console.error("  - " + f));
    process.exit(1);
  }
  console.log("\nAll UI behavioral checks passed.");
  process.exit(0);
}

main().catch((e) => {
  console.error("UI test error:", e.message);
  process.exit(1);
});
