#!/usr/bin/env python3
"""
Shadow-foam auto-calibration.
Black foam top + white foam bottom → bright holes indicate tool slots.
Uses colour contrast + watershed to detect curved shapes, then serves a
web editor where you can rename, adjust vertices, add, and delete slots.
"""
import cv2
import json
import numpy as np
import os
import sys
import time
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

CONFIG_FILE   = "toolchest_config.json"
BASELINE_FILE = "baseline.png"
PORT          = 5001

# ── shared state ──────────────────────────────────────────────────────────────
calib = {
    "phase":       "scan",
    "capture_jpg": None,
    "lock":        threading.Lock(),
}
raw_frame      = None
raw_frame_lock = threading.Lock()
jpg_lock       = threading.Lock()
jpg_buffer     = b""
stop_event     = threading.Event()
cam = cam_mode = None


# ── camera ───────────────────────────────────────────────────────────────────
def open_camera():
    try:
        from picamera2 import Picamera2
        c = Picamera2()
        c.configure(c.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"}))
        c.start()
        time.sleep(1)
        print("[camera] PiCamera2")
        return c, "picamera2"
    except Exception:
        pass
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[camera] OpenCV index {idx}")
            return cap, "opencv"
    print("[ERROR] No camera found.")
    sys.exit(1)


def read_frame(c, mode):
    if mode == "picamera2":
        return cv2.cvtColor(c.capture_array(), cv2.COLOR_RGB2BGR)
    c.grab()
    ret, f = c.retrieve()
    if not ret:
        raise RuntimeError("frame grab failed")
    return f


def release_camera(c, mode):
    if mode == "picamera2": c.stop()
    else: c.release()


def capture_loop():
    global raw_frame, jpg_buffer
    while not stop_event.is_set():
        try:
            frame = read_frame(cam, cam_mode)
            with raw_frame_lock:
                raw_frame = frame
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with jpg_lock:
                jpg_buffer = jpg.tobytes()
        except Exception:
            pass
        time.sleep(0.04)


def mjpeg_stream(handler):
    handler.send_response(200)
    handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
    handler.end_headers()
    try:
        while not stop_event.is_set():
            with jpg_lock:
                jpg = jpg_buffer
            if jpg:
                handler.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
                )
            time.sleep(0.04)
    except Exception:
        pass


# ── detection ────────────────────────────────────────────────────────────────
def detect_foam_slots(frame, threshold=None, min_area=600):
    """
    Finds bright (white-base) regions inside black shadow foam.
    Watershed separates adjacent / touching cutouts into individual slots.
    Returns a list of [[x, y], ...] contour-point arrays.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if threshold is None:
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(blur, int(threshold), 255, cv2.THRESH_BINARY)

    k3 = np.ones((3, 3), np.uint8)
    k7 = np.ones((7, 7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k3, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k7, iterations=2)

    # Distance-transform watershed: peaks become seeds, narrow bridges get cut
    dist   = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_n = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    _, fg  = cv2.threshold(dist_n, 0.5, 1.0, cv2.THRESH_BINARY)
    sure_fg = np.uint8(fg * 255)
    sure_bg = cv2.dilate(binary, k3, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.watershed(frame.copy(), markers)

    result = []
    max_label = int(markers.max())
    for label in range(2, max_label + 1):
        mask = np.uint8(markers == label) * 255
        mask = cv2.dilate(mask, k3, iterations=1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area:
            continue
        # eps=2.0 keeps enough points to represent curves
        approx = cv2.approxPolyDP(c, 2.0, True)
        pts    = approx.reshape(-1, 2).tolist()
        if len(pts) >= 3:
            result.append(pts)
    return result


# ── HTML / JS ────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Foam Calibration</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: sans-serif; background: #0f1115; color: #e2e8f0; padding: 20px; }
h1 { color: #3b82f6; font-size: 1.4em; margin-bottom: 16px; }
.page { display: none; }
.page.active { display: block; }

.instr { background: #1e2330; border: 1px solid #374151; border-radius: 8px;
  padding: 14px 18px; margin-bottom: 18px; color: #94a3b8; line-height: 1.65; }
.instr b { color: #e2e8f0; }

#live-feed { max-width: 800px; width: 100%; border: 2px solid #374151;
  border-radius: 8px; display: block; }

.scan-controls { margin-top: 14px; display: flex; align-items: center;
  gap: 14px; flex-wrap: wrap; }
.thr-group { display: flex; align-items: center; gap: 8px; }
.thr-group label { color: #94a3b8; font-size: 13px; }
input[type=range] { accent-color: #3b82f6; width: 150px; }
#thr-val { width: 28px; text-align: center; font-size: 13px; }
.chk-label { display: flex; align-items: center; gap: 6px;
  color: #94a3b8; font-size: 13px; cursor: pointer; }

.edit-layout { display: flex; gap: 16px; align-items: flex-start; }
#canvas-wrap { flex: 1 1 auto; min-width: 0; }
#draw-canvas { width: 100%; border: 2px solid #374151; border-radius: 8px;
  display: block; cursor: crosshair; }
#draw-canvas.draw-mode { border-color: #eab308; }

.sidebar { width: 260px; flex: 0 0 260px; background: #1e2330;
  border: 1px solid #374151; border-radius: 8px; padding: 14px;
  display: flex; flex-direction: column; gap: 12px; }
.sidebar-head { color: #94a3b8; font-size: 12px; text-transform: uppercase;
  letter-spacing: .08em; }
.slot-list { max-height: 280px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 4px; }
.slot-item { display: flex; align-items: center; gap: 8px; padding: 7px 10px;
  border-radius: 6px; cursor: pointer; border: 1px solid transparent; }
.slot-item:hover { background: #2a3244; }
.slot-item.selected { background: #1d2f52; border-color: #3b82f6; }
.dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.slot-label { font-size: 13px; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
hr { border: none; border-top: 1px solid #374151; }
.field-label { font-size: 12px; color: #94a3b8; margin-bottom: 4px; }
.name-row { display: flex; gap: 6px; }
input[type=text] { background: #111827; color: #e2e8f0; border: 1px solid #374151;
  border-radius: 6px; padding: 7px 10px; font-size: 14px; width: 100%; }
input[type=text]:focus { outline: none; border-color: #3b82f6; }
input[type=text]:disabled { opacity: .4; cursor: not-allowed; }
.hint { font-size: 11px; color: #6b7280; margin-top: 4px; }

.btn { background: #3b82f6; color: #fff; border: none; padding: 8px 14px;
  border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600;
  transition: background .15s; white-space: nowrap; }
.btn:hover  { background: #2563eb; }
.btn:disabled { background: #374151; color: #6b7280; cursor: default; }
.btn-sm { padding: 5px 10px; font-size: 12px; }
.btn-del  { background: #7f1d1d; color: #fca5a5; }
.btn-del:hover  { background: #991b1b; }
.btn-ok   { background: #14532d; color: #86efac; }
.btn-ok:hover   { background: #166534; }
.btn-warn { background: #78350f; color: #fcd34d; }
.btn-warn:hover { background: #92400e; }
.btn-dim  { background: #374151; color: #d1d5db; }
.btn-dim:hover  { background: #4b5563; }
.row { display: flex; gap: 6px; flex-wrap: wrap; }

#draw-banner { display: none; background: #78350f; border: 1px solid #d97706;
  border-radius: 6px; padding: 8px 14px; color: #fcd34d; font-size: 13px;
  margin-bottom: 10px; align-items: center; justify-content: space-between; }
#draw-banner.on { display: flex; }

.done-box { background: #14532d; border: 1px solid #166534; border-radius: 10px;
  padding: 28px; max-width: 480px; }
.done-box h2 { color: #86efac; margin-bottom: 10px; }
.done-box p  { color: #a7f3d0; line-height: 1.6; }
code { background: #1e2330; padding: 2px 6px; border-radius: 4px; color: #93c5fd; }
</style>
</head>
<body>
<h1>&#129520; Tool Chest &mdash; Foam Calibration</h1>

<!-- ── SCAN PAGE ─────────────────────────────────────────────────────────── -->
<div id="page-scan" class="page active">
  <div class="instr">
    <b>Step 1 &mdash; Detect Tool Slots</b><br>
    Remove all tools from the foam so the white base is exposed inside every cutout.<br>
    Leave <b>Auto (Otsu)</b> checked for most setups &mdash; only adjust the threshold if the
    detection misses holes or picks up noise. Then click <b>Detect Slots</b>.
  </div>
  <img id="live-feed" src="/stream">
  <div class="scan-controls">
    <div class="thr-group">
      <label>Threshold:</label>
      <input type="range" id="thr-slider" min="30" max="220" value="128"
             oninput="onSlider(this.value)" disabled>
      <span id="thr-val">&#8212;</span>
    </div>
    <label class="chk-label">
      <input type="checkbox" id="auto-chk" checked onchange="toggleAuto(this.checked)">
      Auto (Otsu)
    </label>
    <button class="btn" id="btn-detect" onclick="runDetect()">Detect Slots</button>
  </div>
</div>

<!-- ── EDIT PAGE ─────────────────────────────────────────────────────────── -->
<div id="page-edit" class="page">
  <div id="draw-banner">
    <span>&#9998;&nbsp;Draw mode &mdash; click to place points (min&nbsp;3).
      Press <kbd>Enter</kbd> or <b>Finish</b> when done, <kbd>Esc</kbd> to cancel.</span>
    <div class="row">
      <button class="btn btn-sm btn-warn" id="btn-finish-draw"
              onclick="finishDraw()" disabled>Finish</button>
      <button class="btn btn-sm btn-dim" onclick="cancelDraw()">Cancel</button>
    </div>
  </div>

  <div class="edit-layout">
    <div id="canvas-wrap">
      <canvas id="draw-canvas"></canvas>
    </div>

    <div class="sidebar">
      <div class="sidebar-head" id="slot-count-lbl">0 slots</div>

      <div class="slot-list" id="slot-list"></div>

      <hr>

      <div>
        <div class="field-label">Selected slot name</div>
        <div class="name-row">
          <input type="text" id="name-inp" placeholder="e.g. pliers" disabled
                 oninput="onNameType(this.value)"
                 onkeydown="if(event.key==='Enter')this.blur()">
          <button class="btn btn-sm btn-del" id="btn-del"
                  onclick="deleteSelected()" disabled title="Delete (Del key)">&#128465;</button>
        </div>
        <div class="hint">Click a slot on the canvas or list to select it.<br>
          Drag its vertex dots to adjust the shape.</div>
      </div>

      <hr>

      <div class="row">
        <button class="btn btn-sm btn-warn" onclick="startDraw()">+ Draw Slot</button>
        <button class="btn btn-sm btn-dim"  onclick="rescan()">&#8635; Re-scan</button>
      </div>

      <hr>

      <button class="btn btn-ok" id="btn-save"
              onclick="saveConfig()" disabled>&#10003; Save Configuration</button>
    </div>
  </div>
</div>

<!-- ── DONE PAGE ──────────────────────────────────────────────────────────── -->
<div id="page-done" class="page">
  <div class="done-box">
    <h2>&#10003; Calibration Complete</h2>
    <p>Saved <span id="done-count">0</span> tool slots to
      <code>toolchest_config.json</code>.<br><br>
      Close this window and start <code>detect.py</code>.</p>
  </div>
</div>

<script>
// ── colour palette ──────────────────────────────────────────────────────────
const STROKE = ['#ef4444','#f97316','#facc15','#4ade80','#22d3ee',
                '#a78bfa','#f472b6','#34d399','#fb923c','#60a5fa'];
const FILL   = STROKE.map(c => c + '30');
const FILL_S = STROKE.map(c => c + '55');

// ── state ───────────────────────────────────────────────────────────────────
let slots      = [];   // [{id, name, points:[[x,y],...]}]
let nextId     = 0;
let selId      = null;
let drawMode   = false;
let drawPts    = [];
let dragInfo   = null; // {slotId, ptIdx}
let baseImg    = null;

const canvas   = document.getElementById("draw-canvas");
const ctx      = canvas.getContext("2d");

// ── page nav ────────────────────────────────────────────────────────────────
function showPage(n) {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.getElementById("page-" + n).classList.add("active");
}

// ── geometry helpers ─────────────────────────────────────────────────────────
function makePath(pts) {
  const p = new Path2D();
  pts.forEach((q, i) => i === 0 ? p.moveTo(q[0], q[1]) : p.lineTo(q[0], q[1]));
  p.closePath();
  return p;
}

function centroid(pts) {
  return [
    Math.round(pts.reduce((s, p) => s + p[0], 0) / pts.length),
    Math.round(pts.reduce((s, p) => s + p[1], 0) / pts.length)
  ];
}

function canvasXY(e) {
  const r  = canvas.getBoundingClientRect();
  return [
    Math.round((e.clientX - r.left) * canvas.width  / r.width),
    Math.round((e.clientY - r.top)  * canvas.height / r.height)
  ];
}

function ci(slotId) {
  const i = slots.findIndex(s => s.id === slotId);
  return (i < 0 ? 0 : i) % STROKE.length;
}

// ── render ───────────────────────────────────────────────────────────────────
function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (baseImg) ctx.drawImage(baseImg, 0, 0);

  slots.forEach(s => {
    if (s.points.length < 2) return;
    const c   = ci(s.id);
    const sel = s.id === selId;
    const p   = makePath(s.points);

    ctx.fillStyle   = sel ? FILL_S[c] : FILL[c];
    ctx.fill(p);
    ctx.strokeStyle = STROKE[c];
    ctx.lineWidth   = sel ? 3 : 2;
    ctx.setLineDash([]);
    ctx.stroke(p);

    const [cx, cy] = centroid(s.points);
    ctx.font        = "bold 13px sans-serif";
    ctx.textAlign   = "center";
    ctx.shadowColor = "#000"; ctx.shadowBlur = 5;
    ctx.fillStyle   = "#fff";
    ctx.fillText(s.name || "?", cx, cy + 5);
    ctx.shadowBlur  = 0;
    ctx.textAlign   = "left";

    if (sel) {
      s.points.forEach(q => {
        ctx.beginPath();
        ctx.arc(q[0], q[1], 5, 0, 2 * Math.PI);
        ctx.fillStyle   = "#fff";
        ctx.fill();
        ctx.strokeStyle = STROKE[c];
        ctx.lineWidth   = 2;
        ctx.stroke();
      });
    }
  });

  if (drawMode && drawPts.length > 0) {
    ctx.beginPath();
    drawPts.forEach((p, i) => i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]));
    ctx.strokeStyle = "#facc15";
    ctx.lineWidth   = 2;
    ctx.setLineDash([8, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
    drawPts.forEach((p, i) => {
      ctx.beginPath();
      ctx.arc(p[0], p[1], i === 0 ? 7 : 5, 0, 2 * Math.PI);
      ctx.fillStyle = i === 0 ? "#facc15" : "#fff";
      ctx.fill();
    });
  }
}

// ── slot list ─────────────────────────────────────────────────────────────────
function renderList() {
  document.getElementById("slot-list").innerHTML = slots.map((s, i) =>
    `<div class="slot-item ${s.id === selId ? 'selected' : ''}" onclick="select(${s.id})">
       <span class="dot" style="background:${STROKE[i % STROKE.length]}"></span>
       <span class="slot-label">${s.name || 'unnamed'}</span>
     </div>`
  ).join("");
  const n = slots.length;
  document.getElementById("slot-count-lbl").textContent =
    n + " slot" + (n !== 1 ? "s" : "");
  document.getElementById("btn-save").disabled = n === 0;
}

function select(id) {
  selId = id;
  const inp = document.getElementById("name-inp");
  const del = document.getElementById("btn-del");
  if (id !== null) {
    const s = slots.find(s => s.id === id);
    inp.value = s ? (s.name || "") : "";
    inp.disabled = false;
    del.disabled = false;
  } else {
    inp.value = ""; inp.disabled = true; del.disabled = true;
  }
  renderList();
  redraw();
}

function onNameType(v) {
  if (selId === null) return;
  const s = slots.find(s => s.id === selId);
  if (s) { s.name = v; renderList(); redraw(); }
}

// ── hit testing ──────────────────────────────────────────────────────────────
function hitSlot(x, y) {
  for (let i = slots.length - 1; i >= 0; i--)
    if (ctx.isPointInPath(makePath(slots[i].points), x, y)) return slots[i].id;
  return null;
}

function hitVertex(x, y, r = 12) {
  if (selId === null) return null;
  const s = slots.find(s => s.id === selId);
  if (!s) return null;
  for (let i = 0; i < s.points.length; i++)
    if (Math.hypot(s.points[i][0] - x, s.points[i][1] - y) < r) return i;
  return null;
}

// ── canvas events ────────────────────────────────────────────────────────────
canvas.addEventListener("mousedown", e => {
  const [cx, cy] = canvasXY(e);
  if (drawMode) {
    drawPts.push([cx, cy]);
    document.getElementById("btn-finish-draw").disabled = drawPts.length < 3;
    redraw(); return;
  }
  const vi = hitVertex(cx, cy);
  if (vi !== null) { dragInfo = { slotId: selId, ptIdx: vi }; return; }
  select(hitSlot(cx, cy));
});

canvas.addEventListener("mousemove", e => {
  if (!dragInfo) return;
  const [cx, cy] = canvasXY(e);
  const s = slots.find(s => s.id === dragInfo.slotId);
  if (s) { s.points[dragInfo.ptIdx] = [cx, cy]; redraw(); }
});

canvas.addEventListener("mouseup",    () => { dragInfo = null; });
canvas.addEventListener("mouseleave", () => { dragInfo = null; });

document.addEventListener("keydown", e => {
  const active = document.activeElement;
  const typing = active && (active.tagName === "INPUT");
  if ((e.key === "Delete" || e.key === "Backspace") && !typing && selId !== null) {
    e.preventDefault(); deleteSelected();
  }
  if (e.key === "Escape") { drawMode ? cancelDraw() : select(null); }
  if (e.key === "Enter"  && drawMode) finishDraw();
});

// ── operations ───────────────────────────────────────────────────────────────
function deleteSelected() {
  slots = slots.filter(s => s.id !== selId);
  select(null);
  renderList();
  redraw();
}

function startDraw() {
  drawMode = true; drawPts = [];
  canvas.classList.add("draw-mode");
  document.getElementById("draw-banner").classList.add("on");
  document.getElementById("btn-finish-draw").disabled = true;
  select(null);
}

function finishDraw() {
  if (drawPts.length < 3) return;
  slots.push({ id: nextId++, name: "tool_" + (slots.length + 1), points: [...drawPts] });
  cancelDraw();
  select(slots[slots.length - 1].id);
  renderList();
}

function cancelDraw() {
  drawMode = false; drawPts = [];
  canvas.classList.remove("draw-mode");
  document.getElementById("draw-banner").classList.remove("on");
  redraw();
}

// ── scan page ────────────────────────────────────────────────────────────────
function toggleAuto(on) {
  document.getElementById("thr-slider").disabled = on;
  document.getElementById("thr-val").textContent = on
    ? "—" : document.getElementById("thr-slider").value;
}

function onSlider(v) {
  document.getElementById("thr-val").textContent = v;
}

async function runDetect() {
  const btn = document.getElementById("btn-detect");
  btn.disabled = true; btn.textContent = "Detecting…";
  const isAuto = document.getElementById("auto-chk").checked;
  const thr    = isAuto ? null : parseInt(document.getElementById("thr-slider").value);
  try {
    const res  = await fetch("/api/detect", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ threshold: thr })
    });
    const data = await res.json();
    if (!res.ok) { alert("Detection failed: " + (data.error || "unknown")); return; }

    baseImg = new Image();
    await new Promise((ok, fail) => {
      baseImg.onload = ok; baseImg.onerror = fail;
      baseImg.src = "/capture.jpg?" + Date.now();
    });
    canvas.width  = baseImg.naturalWidth;
    canvas.height = baseImg.naturalHeight;

    slots  = data.slots.map((pts, i) => ({ id: nextId++, name: "tool_" + (i + 1), points: pts }));
    selId  = null;
    renderList();
    redraw();
    showPage("edit");
  } catch (err) {
    alert("Error: " + err);
  } finally {
    btn.disabled = false; btn.textContent = "Detect Slots";
  }
}

function rescan() {
  if (slots.length > 0 && !confirm("Re-scan will replace all current slots. Continue?")) return;
  showPage("scan");
}

// ── save ─────────────────────────────────────────────────────────────────────
async function saveConfig() {
  const payload = slots.map(s => ({ name: s.name || ("tool_" + s.id), points: s.points }));
  const res  = await fetch("/api/save", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ slots: payload })
  });
  const data = await res.json();
  if (!res.ok) { alert("Save failed: " + (data.error || "unknown")); return; }
  document.getElementById("done-count").textContent = data.count;
  showPage("done");
}
</script>
</body>
</html>"""


# ── HTTP handler ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/stream":
            mjpeg_stream(self)

        elif self.path.startswith("/capture.jpg"):
            with calib["lock"]:
                jpg = calib["capture_jpg"]
            if jpg:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
            else:
                self.send_response(404)
                self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        cl   = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(cl) if cl else b""

        if self.path == "/api/detect":
            data  = json.loads(body) if body else {}
            thr   = data.get("threshold")
            with raw_frame_lock:
                frame = raw_frame
            if frame is None:
                self.send_json(500, {"error": "No camera frame yet"}); return

            cv2.imwrite(BASELINE_FILE, frame)
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with calib["lock"]:
                calib["capture_jpg"] = jpg.tobytes()

            detected = detect_foam_slots(frame, thr)
            self.send_json(200, {"slots": detected, "count": len(detected)})

        elif self.path == "/api/save":
            data     = json.loads(body)
            slots_in = data.get("slots", [])
            if not slots_in:
                self.send_json(400, {"error": "No slots defined"}); return

            slots_out = []
            for i, s in enumerate(slots_in):
                pts = [[int(p[0]), int(p[1])] for p in s["points"]]
                xs  = [p[0] for p in pts]
                ys  = [p[1] for p in pts]
                slots_out.append({
                    "name":                (s.get("name") or f"tool_{i+1}").strip(),
                    "x":                   min(xs),
                    "y":                   min(ys),
                    "w":                   max(xs) - min(xs),
                    "h":                   max(ys) - min(ys),
                    "polygon_pts":         pts,
                    "baseline_dark_ratio": 0.0,
                })

            config = {
                "created":        datetime.now().isoformat(),
                "baseline_image": BASELINE_FILE,
                "slots":          slots_out,
                "settings": {
                    "detection_mode":      "background_subtract",
                    "bg_diff_threshold":   30,
                    "present_pixel_ratio": 0.05,
                },
            }
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
            with calib["lock"]:
                calib["phase"] = "done"
            self.send_json(200, {"ok": True, "count": len(slots_out)})
            stop_event.set()

        else:
            self.send_response(404)
            self.end_headers()


# ── main ──────────────────────────────────────────────────────────────────────
class ThreadedHTTP(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    global cam, cam_mode
    cam, cam_mode = open_camera()
    threading.Thread(target=capture_loop, daemon=True).start()

    server = ThreadedHTTP(("0.0.0.0", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[calibrate] Open  http://localhost:{PORT}  to begin.")

    try:
        stop_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        release_camera(cam, cam_mode)
        if calib["phase"] == "done":
            print(f"[calibrate] Configuration saved to {CONFIG_FILE}.")
        else:
            print("[calibrate] Cancelled — nothing saved.")


if __name__ == "__main__":
    main()
