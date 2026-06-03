#!/usr/bin/env python3
"""
Shadow-foam auto-calibration.
Black foam top + white foam bottom → bright holes indicate tool slots.
Features: auto-detect, foam boundary, split/join line, vertex drag, rename/delete.
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
    "frame_size":  (1280, 720),
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
        c.start(); time.sleep(1)
        print("[camera] PiCamera2"); return c, "picamera2"
    except Exception: pass
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[camera] OpenCV index {idx}"); return cap, "opencv"
    print("[ERROR] No camera found."); sys.exit(1)


def read_frame(c, mode):
    if mode == "picamera2":
        return cv2.cvtColor(c.capture_array(), cv2.COLOR_RGB2BGR)
    c.grab(); ret, f = c.retrieve()
    if not ret: raise RuntimeError("frame grab failed")
    return f


def release_camera(c, mode):
    if mode == "picamera2": c.stop()
    else: c.release()


def capture_loop():
    global raw_frame, jpg_buffer
    while not stop_event.is_set():
        try:
            frame = read_frame(cam, cam_mode)
            with raw_frame_lock: raw_frame = frame
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with jpg_lock: jpg_buffer = jpg.tobytes()
        except Exception: pass
        time.sleep(0.04)


def mjpeg_stream(handler):
    handler.send_response(200)
    handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
    handler.end_headers()
    try:
        while not stop_event.is_set():
            with jpg_lock: jpg = jpg_buffer
            if jpg:
                handler.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
                )
            time.sleep(0.04)
    except Exception: pass


# ── detection ────────────────────────────────────────────────────────────────
def detect_foam_slots(frame, threshold=None, min_area=600, boundary_pts=None):
    """
    Detects bright holes in black shadow foam via contrast + watershed.
    Returns list of [[x, y], ...] contour-point arrays.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if threshold is None:
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(blur, int(threshold), 255, cv2.THRESH_BINARY)
    k3 = np.ones((3, 3), np.uint8); k7 = np.ones((7, 7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k3, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k7, iterations=2)
    if boundary_pts and len(boundary_pts) >= 3:
        bmask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.fillPoly(bmask, [np.array(boundary_pts, np.int32)], 255)
        binary = cv2.bitwise_and(binary, bmask)

    dist   = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_n = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    _, fg  = cv2.threshold(dist_n, 0.5, 1.0, cv2.THRESH_BINARY)
    sure_fg = np.uint8(fg * 255)
    sure_bg = cv2.dilate(binary, k3, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1; markers[unknown == 255] = 0
    cv2.watershed(frame.copy(), markers)

    result = []
    for label in range(2, int(markers.max()) + 1):
        mask = np.uint8(markers == label) * 255
        mask = cv2.dilate(mask, k3, iterations=1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts: continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area: continue
        approx = cv2.approxPolyDP(c, 2.0, True)
        pts    = approx.reshape(-1, 2).tolist()
        if len(pts) >= 3: result.append(pts)
    return result


# ── split / join helpers ──────────────────────────────────────────────────────
def _pt_in_slot(px, py, slot_pts, fw, fh):
    mask = np.zeros((fh, fw), np.uint8)
    cv2.fillPoly(mask, [np.array(slot_pts, np.int32)], 255)
    px, py = int(np.clip(px, 0, fw - 1)), int(np.clip(py, 0, fh - 1))
    return bool(mask[py, px])


def _contour_from_mask(m, min_area=100):
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area: return None
    return cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2).tolist()


def split_slot_by_line(pts, p1, p2, fw, fh):
    """Split one polygon along a line. Returns list of 0-2 new polygon lists."""
    mask = np.zeros((fh, fw), np.uint8)
    cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
    Y, X   = np.mgrid[0:fh, 0:fw]
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    cross  = dx * (Y.astype(np.int64) - p1[1]) - dy * (X.astype(np.int64) - p1[0])
    ha = np.zeros((fh, fw), np.uint8); ha[cross >= 0] = 255
    hb = np.zeros((fh, fw), np.uint8); hb[cross <  0] = 255
    result = []
    for h in (ha, hb):
        part = cv2.bitwise_and(mask, h)
        c    = _contour_from_mask(part)
        if c: result.append(c)
    return result


def join_slots_by_line(pts_a, pts_b, p1, p2, fw, fh):
    """Merge two polygons, bridging along the drawn line. Returns one polygon list."""
    mask = np.zeros((fh, fw), np.uint8)
    cv2.fillPoly(mask, [np.array(pts_a, np.int32)], 255)
    cv2.fillPoly(mask, [np.array(pts_b, np.int32)], 255)
    bridge = max(8, int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]) * 0.25))
    cv2.line(mask, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 255, bridge)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    c = _contour_from_mask(mask)
    return [c] if c else [pts_a, pts_b]


def process_line(p1, p2, slots_in, fw, fh):
    """
    Determine if line splits (both endpoints/midpoint in same slot)
    or joins (endpoints/midpoint in different slots), apply the operation.
    Returns (message, removed_ids, added_slots).
    """
    samples = [p1, [(p1[0]+p2[0])//2, (p1[1]+p2[1])//2], p2]
    hits = {}   # slot id → slot dict
    for sp in samples:
        for s in slots_in:
            if s["id"] not in hits and _pt_in_slot(sp[0], sp[1], s["points"], fw, fh):
                hits[s["id"]] = s
    ids = list(hits.keys())

    if len(ids) == 1:
        s      = hits[ids[0]]
        parts  = split_slot_by_line(s["points"], p1, p2, fw, fh)
        if len(parts) < 2:
            return "Line doesn't divide the slot cleanly", [], []
        added  = [{"name": s["name"] + "_1", "points": parts[0]},
                  {"name": s["name"] + "_2", "points": parts[1]}]
        return f"Split '{s['name']}'", [s["id"]], added

    if len(ids) >= 2:
        a, b   = hits[ids[0]], hits[ids[1]]
        merged = join_slots_by_line(a["points"], b["points"], p1, p2, fw, fh)
        added  = [{"name": a["name"] + "+" + b["name"], "points": p} for p in merged]
        return f"Joined '{a['name']}' and '{b['name']}'", [a["id"], b["id"]], added

    return "Line did not intersect any slots", [], []


# ── HTML ──────────────────────────────────────────────────────────────────────
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
.scan-row { margin-top: 14px; display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
.thr-group { display: flex; align-items: center; gap: 8px; }
.thr-group label { color: #94a3b8; font-size: 13px; }
input[type=range] { accent-color: #3b82f6; width: 150px; }
#thr-val { width: 28px; text-align: center; font-size: 13px; }
.chk-label { display: flex; align-items: center; gap: 6px;
  color: #94a3b8; font-size: 13px; cursor: pointer; user-select: none; }

/* edit layout */
.edit-layout { display: flex; gap: 16px; align-items: flex-start; }
#canvas-wrap { flex: 1 1 auto; min-width: 0; }
#draw-canvas { width: 100%; border: 2px solid #374151; border-radius: 8px;
  display: block; cursor: crosshair; }
.sidebar { width: 260px; flex: 0 0 260px; background: #1e2330;
  border: 1px solid #374151; border-radius: 8px; padding: 14px;
  display: flex; flex-direction: column; gap: 11px; }
.sidebar-lbl { color: #94a3b8; font-size: 11px; text-transform: uppercase;
  letter-spacing: .08em; }
.slot-list { max-height: 260px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 3px; }
.slot-item { display: flex; align-items: center; gap: 8px; padding: 6px 10px;
  border-radius: 6px; cursor: pointer; border: 1px solid transparent; }
.slot-item:hover { background: #2a3244; }
.slot-item.sel { background: #1d2f52; border-color: #3b82f6; }
.dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.slot-name-lbl { font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
hr { border: none; border-top: 1px solid #2d3748; }
.name-row { display: flex; gap: 6px; }
input[type=text] { background: #111827; color: #e2e8f0; border: 1px solid #374151;
  border-radius: 6px; padding: 7px 10px; font-size: 14px; width: 100%; }
input[type=text]:focus { outline: none; border-color: #3b82f6; }
input[type=text]:disabled { opacity: .4; cursor: not-allowed; }
.hint { font-size: 11px; color: #6b7280; }

.btn { background: #3b82f6; color: #fff; border: none; padding: 8px 14px;
  border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600;
  transition: background .15s; white-space: nowrap; }
.btn:hover  { background: #2563eb; }
.btn:disabled { background: #374151 !important; color: #6b7280 !important; cursor: default; }
.btn-sm { padding: 5px 10px; font-size: 12px; }
.btn-del  { background: #7f1d1d; color: #fca5a5; } .btn-del:hover  { background: #991b1b; }
.btn-ok   { background: #14532d; color: #86efac; } .btn-ok:hover   { background: #166534; }
.btn-warn { background: #78350f; color: #fcd34d; } .btn-warn:hover { background: #92400e; }
.btn-dim  { background: #374151; color: #d1d5db; } .btn-dim:hover  { background: #4b5563; }
.row { display: flex; gap: 6px; flex-wrap: wrap; }

#banner { display: none; background: #1c1a08; border: 1px solid #d97706;
  border-radius: 6px; padding: 9px 14px; color: #fcd34d; font-size: 13px;
  margin-bottom: 10px; align-items: center; justify-content: space-between; gap: 8px; }
#banner.on { display: flex; }
#banner kbd { background: #374151; border: 1px solid #4b5563; border-radius: 3px;
  padding: 1px 5px; font-size: 11px; color: #d1d5db; }

.done-box { background: #14532d; border: 1px solid #166534; border-radius: 10px;
  padding: 28px; max-width: 480px; }
.done-box h2 { color: #86efac; margin-bottom: 10px; }
.done-box p  { color: #a7f3d0; line-height: 1.6; }
code { background: #1e2330; padding: 2px 6px; border-radius: 4px; color: #93c5fd; }

#toast { position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%) translateY(20px);
  background: #1e3a5f; border: 1px solid #3b82f6; color: #93c5fd; padding: 9px 18px;
  border-radius: 8px; font-size: 13px; opacity: 0; transition: opacity .25s, transform .25s;
  pointer-events: none; z-index: 999; }
#toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
</style>
</head>
<body>
<h1>&#129520; Tool Chest &mdash; Foam Calibration</h1>

<!-- SCAN -->
<div id="page-scan" class="page active">
  <div class="instr">
    <b>Step 1 &mdash; Detect Tool Slots</b><br>
    Remove all tools from the foam so the white base is visible inside every cutout.<br>
    Leave <b>Auto (Otsu)</b> checked for most setups. Click <b>Detect Slots</b> when ready.
  </div>
  <img id="live-feed" src="/stream">
  <div class="scan-row">
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

<!-- EDIT -->
<div id="page-edit" class="page">
  <div id="banner">
    <span id="banner-msg"></span>
    <div class="row">
      <button class="btn btn-sm btn-warn" id="btn-finish" onclick="finishDraw()" style="display:none">Finish</button>
      <button class="btn btn-sm btn-dim"  id="btn-cancel" onclick="cancelDraw()">Cancel</button>
    </div>
  </div>

  <div class="edit-layout">
    <div id="canvas-wrap">
      <canvas id="draw-canvas"></canvas>
    </div>

    <div class="sidebar">
      <div class="sidebar-lbl" id="slot-count-lbl">0 slots</div>
      <div class="slot-list" id="slot-list"></div>
      <hr>

      <div>
        <div class="sidebar-lbl" style="margin-bottom:6px;">Selected slot</div>
        <div class="name-row">
          <input type="text" id="name-inp" placeholder="e.g. pliers" disabled
                 oninput="onNameType(this.value)"
                 onkeydown="if(event.key==='Enter')this.blur()">
          <button class="btn btn-sm btn-del" id="btn-del"
                  onclick="deleteSelected()" disabled title="Delete (Del)">&#128465;</button>
        </div>
        <div class="hint" style="margin-top:5px;">
          Click to select &bull; Drag vertex to reshape &bull; Right-click vertex to delete it
        </div>
      </div>
      <hr>

      <div class="sidebar-lbl">Tools</div>
      <div class="row">
        <button class="btn btn-sm btn-warn" id="btn-new-slot"  onclick="startSlot()">+ Slot</button>
        <button class="btn btn-sm"          id="btn-line"      onclick="startLine()">&#9986; Split / Join</button>
      </div>
      <div class="row">
        <button class="btn btn-sm btn-dim"  id="btn-bdry"      onclick="startBdry()">&#9645; Set Boundary</button>
        <button class="btn btn-sm btn-del"  id="btn-clr-bdry"  onclick="clearBdry()" disabled>&#10005; Boundary</button>
      </div>
      <hr>

      <div class="row">
        <button class="btn btn-sm btn-dim" id="btn-rescan" onclick="rescan()">&#8635; Re-scan</button>
      </div>
      <hr>

      <button class="btn btn-ok" id="btn-save" onclick="saveConfig()" disabled>
        &#10003; Save Configuration
      </button>
    </div>
  </div>
</div>

<!-- DONE -->
<div id="page-done" class="page">
  <div class="done-box">
    <h2>&#10003; Calibration Complete</h2>
    <p>Saved <span id="done-count">0</span> tool slots to
      <code>toolchest_config.json</code>.<br><br>
      Close this window and start <code>detect.py</code>.</p>
  </div>
</div>

<div id="toast"></div>

<script>
// ── palette ─────────────────────────────────────────────────────────────────
const STROKE = ['#ef4444','#f97316','#facc15','#4ade80','#22d3ee',
                '#a78bfa','#f472b6','#34d399','#fb923c','#60a5fa'];
const FILL   = STROKE.map(c => c + '28');
const FILL_S = STROKE.map(c => c + '50');

// ── canvas + image ──────────────────────────────────────────────────────────
const canvas = document.getElementById("draw-canvas");
const ctx    = canvas.getContext("2d");
let baseImg  = null;

// ── slots state ──────────────────────────────────────────────────────────────
let slots  = [];
let nextId = 0;
let selId  = null;
let dragInfo = null;   // {slotId, ptIdx}

// ── draw mode state ──────────────────────────────────────────────────────────
let mode     = null;   // null | 'slot' | 'bdry' | 'line'
let drawPts  = [];     // points for slot/boundary polygon in progress
let linePt1  = null;   // first point for split/join line
let mousePos = null;   // live cursor position (previews)

// ── boundary ─────────────────────────────────────────────────────────────────
let boundary = null;   // [[x,y],...] or null

// ── helpers ──────────────────────────────────────────────────────────────────
function showPage(n) {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.getElementById("page-" + n).classList.add("active");
}

function makePath(pts) {
  const p = new Path2D();
  pts.forEach((q, i) => i === 0 ? p.moveTo(q[0], q[1]) : p.lineTo(q[0], q[1]));
  p.closePath(); return p;
}
function centroid(pts) {
  return [
    Math.round(pts.reduce((s, p) => s + p[0], 0) / pts.length),
    Math.round(pts.reduce((s, p) => s + p[1], 0) / pts.length)
  ];
}
function canvasXY(e) {
  const r = canvas.getBoundingClientRect();
  return [
    Math.round((e.clientX - r.left) * canvas.width  / r.width),
    Math.round((e.clientY - r.top)  * canvas.height / r.height)
  ];
}
function ci(id) {
  const i = slots.findIndex(s => s.id === id);
  return (i < 0 ? 0 : i) % STROKE.length;
}
function isInsidePoly([px, py], poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > py) !== (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi)
      inside = !inside;
  }
  return inside;
}

// ── toast ────────────────────────────────────────────────────────────────────
let _toastTimer = null;
function toast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg; el.classList.add("show");
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove("show"), 3000);
}

// ── redraw ───────────────────────────────────────────────────────────────────
function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (baseImg) ctx.drawImage(baseImg, 0, 0);

  // Confirmed slots
  slots.forEach(s => {
    if (s.points.length < 2) return;
    const c   = ci(s.id);
    const sel = s.id === selId;
    const p   = makePath(s.points);
    ctx.fillStyle = sel ? FILL_S[c] : FILL[c]; ctx.fill(p);
    ctx.strokeStyle = STROKE[c]; ctx.lineWidth = sel ? 3 : 1.5;
    ctx.setLineDash([]); ctx.stroke(p);
    const [cx, cy] = centroid(s.points);
    ctx.font = "bold 13px sans-serif"; ctx.textAlign = "center";
    ctx.shadowColor = "#000"; ctx.shadowBlur = 5;
    ctx.fillStyle = "#fff"; ctx.fillText(s.name || "?", cx, cy + 5);
    ctx.shadowBlur = 0; ctx.textAlign = "left";
    if (sel) s.points.forEach(q => {
      ctx.beginPath(); ctx.arc(q[0], q[1], 5, 0, 2*Math.PI);
      ctx.fillStyle = "#fff"; ctx.fill();
      ctx.strokeStyle = STROKE[c]; ctx.lineWidth = 2; ctx.stroke();
    });
  });

  // Confirmed boundary
  if (boundary && boundary.length >= 3) {
    ctx.beginPath();
    boundary.forEach((p, i) => i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]));
    ctx.closePath();
    ctx.strokeStyle = "rgba(255,255,255,0.75)"; ctx.lineWidth = 2;
    ctx.setLineDash([14, 6]); ctx.stroke(); ctx.setLineDash([]);
  }

  // In-progress polygon (slot or boundary)
  if ((mode === 'slot' || mode === 'bdry') && drawPts.length > 0) {
    const color = mode === 'slot' ? "#facc15" : "rgba(255,255,255,0.9)";
    ctx.beginPath();
    drawPts.forEach((p, i) => i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]));
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.setLineDash(mode === 'slot' ? [8, 4] : [14, 6]); ctx.stroke(); ctx.setLineDash([]);
    drawPts.forEach((p, i) => {
      ctx.beginPath(); ctx.arc(p[0], p[1], i === 0 ? 7 : 5, 0, 2*Math.PI);
      ctx.fillStyle = color; ctx.fill();
    });
  }

  // Split/join line
  if (mode === 'line' && linePt1) {
    ctx.beginPath(); ctx.moveTo(linePt1[0], linePt1[1]);
    if (mousePos) ctx.lineTo(mousePos[0], mousePos[1]);
    ctx.strokeStyle = "#f97316"; ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]); ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(linePt1[0], linePt1[1], 6, 0, 2*Math.PI);
    ctx.fillStyle = "#f97316"; ctx.fill();
  }
}

// ── slot list rendering ───────────────────────────────────────────────────────
function renderList() {
  document.getElementById("slot-list").innerHTML = slots.map((s, i) =>
    `<div class="slot-item ${s.id === selId ? 'sel' : ''}" onclick="pick(${s.id})">
       <span class="dot" style="background:${STROKE[i % STROKE.length]}"></span>
       <span class="slot-name-lbl">${s.name || 'unnamed'}</span>
     </div>`
  ).join("");
  const n = slots.length;
  document.getElementById("slot-count-lbl").textContent = n + " slot" + (n !== 1 ? "s" : "");
}

function pick(id) {
  selId = id;
  const inp = document.getElementById("name-inp");
  const del = document.getElementById("btn-del");
  if (id !== null) {
    const s = slots.find(s => s.id === id);
    inp.value = s ? (s.name || "") : "";
    inp.disabled = mode !== null; del.disabled = mode !== null;
  } else {
    inp.value = ""; inp.disabled = true; del.disabled = true;
  }
  renderList(); redraw();
}
function onNameType(v) {
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

// ── canvas events ─────────────────────────────────────────────────────────────
canvas.addEventListener("mousedown", e => {
  const [cx, cy] = canvasXY(e);

  if (mode === 'slot' || mode === 'bdry') {
    drawPts.push([cx, cy]);
    document.getElementById("btn-finish").disabled = drawPts.length < 3;
    redraw(); return;
  }
  if (mode === 'line') {
    if (!linePt1) { linePt1 = [cx, cy]; updateBanner(); redraw(); }
    else          { applyLine(linePt1, [cx, cy]); }
    return;
  }
  const vi = hitVertex(cx, cy);
  if (vi !== null) { dragInfo = { slotId: selId, ptIdx: vi }; return; }
  pick(hitSlot(cx, cy));
});

canvas.addEventListener("mousemove", e => {
  const [cx, cy] = canvasXY(e);
  if (dragInfo) {
    const s = slots.find(s => s.id === dragInfo.slotId);
    if (s) { s.points[dragInfo.ptIdx] = [cx, cy]; redraw(); } return;
  }
  if ((mode === 'line' && linePt1) || mode === 'slot' || mode === 'bdry') {
    mousePos = [cx, cy]; redraw();
  }
});
canvas.addEventListener("mouseup",    () => { dragInfo = null; });
canvas.addEventListener("mouseleave", () => { dragInfo = null; });

canvas.addEventListener("contextmenu", e => {
  e.preventDefault();
  if (mode !== null || selId === null) return;
  const [cx, cy] = canvasXY(e);
  const vi = hitVertex(cx, cy);
  if (vi === null) return;
  const s = slots.find(s => s.id === selId);
  if (!s) return;
  if (s.points.length <= 3) {
    toast("Need at least 3 vertices — use \u{1F5D1} to delete the whole slot");
    return;
  }
  s.points.splice(vi, 1);
  redraw();
});

document.addEventListener("keydown", e => {
  const typing = document.activeElement && document.activeElement.tagName === "INPUT";
  if ((e.key === "Delete" || e.key === "Backspace") && !typing && selId !== null && mode === null) {
    e.preventDefault(); deleteSelected();
  }
  if (e.key === "Escape") { if (mode) cancelDraw(); else pick(null); }
  if (e.key === "Enter" && (mode === 'slot' || mode === 'bdry') && drawPts.length >= 3)
    finishDraw();
});

// ── draw mode management ──────────────────────────────────────────────────────
function setMode(m) {
  mode = m; drawPts = []; linePt1 = null; mousePos = null;
  // Disable/enable buttons
  const active = m !== null;
  ["btn-new-slot","btn-line","btn-bdry","btn-rescan"].forEach(id =>
    document.getElementById(id).disabled = active
  );
  document.getElementById("btn-clr-bdry").disabled = active || !boundary;
  document.getElementById("btn-save").disabled     = active || slots.length === 0;
  document.getElementById("btn-del").disabled      = active || selId === null;
  document.getElementById("name-inp").disabled     = active || selId === null;
  updateBanner();
}

function updateBanner() {
  const banner  = document.getElementById("banner");
  const msg     = document.getElementById("banner-msg");
  const finBtn  = document.getElementById("btn-finish");
  if (!mode) { banner.classList.remove("on"); return; }
  banner.classList.add("on");
  finBtn.style.display = mode === 'line' ? "none" : "inline-block";
  finBtn.disabled      = drawPts.length < 3;
  if (mode === 'slot') msg.innerHTML =
    `&#9998; <b>New slot:</b> click to place points (min&nbsp;3). <kbd>Enter</kbd> or <b>Finish</b> when done.`;
  else if (mode === 'bdry') msg.innerHTML =
    `&#9645; <b>Boundary:</b> trace the foam edge (min&nbsp;3 points). <kbd>Enter</kbd> or <b>Finish</b> when done.`;
  else if (mode === 'line') msg.innerHTML = linePt1
    ? `&#9986; <b>Split/Join:</b> click the <b>second</b> point.`
    : `&#9986; <b>Split/Join:</b> click the <b>first</b> point. Draw as many lines as needed &mdash; <kbd>Esc</kbd> to exit.`;
}

function startSlot()  { pick(null); setMode('slot');  }
function startBdry()  { pick(null); setMode('bdry');  }
function startLine()  { pick(null); setMode('line');  }

function finishDraw() {
  if (mode === 'slot') {
    if (drawPts.length < 3) return;
    slots.push({ id: nextId++, name: "tool_" + (slots.length + 1), points: [...drawPts] });
    setMode(null);
    pick(slots[slots.length - 1].id);
    renderList();
  } else if (mode === 'bdry') {
    if (drawPts.length < 3) return;
    boundary = [...drawPts];
    document.getElementById("btn-clr-bdry").disabled = false;
    toast("Boundary set — slots outside it will be filtered on re-scan");
    setMode(null); redraw();
  }
}

function cancelDraw() { setMode(null); redraw(); }

function deleteSelected() {
  slots = slots.filter(s => s.id !== selId);
  pick(null); renderList(); redraw();
}
function clearBdry()  {
  boundary = null;
  document.getElementById("btn-clr-bdry").disabled = true;
  redraw(); toast("Boundary cleared");
}

// ── split / join line ─────────────────────────────────────────────────────────
async function applyLine(p1, p2) {
  // Reset for next line — stay in line mode so user can draw another immediately
  linePt1 = null; mousePos = null; updateBanner(); redraw();
  const payload = {
    slots: slots.map(s => ({ id: s.id, name: s.name, points: s.points })),
    line:  [p1, p2]
  };
  const res  = await fetch("/api/split_join", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  if (data.action === "none") { toast(data.message); return; }
  data.removed_ids.forEach(id => { slots = slots.filter(s => s.id !== id); });
  data.added.forEach(s => slots.push({ id: nextId++, name: s.name, points: s.points }));
  pick(null); renderList(); redraw();
  toast(data.message);
}

// ── scan page ─────────────────────────────────────────────────────────────────
function toggleAuto(on) {
  document.getElementById("thr-slider").disabled = on;
  document.getElementById("thr-val").textContent = on
    ? "—" : document.getElementById("thr-slider").value;
}
function onSlider(v) { document.getElementById("thr-val").textContent = v; }

async function runDetect() {
  const btn = document.getElementById("btn-detect");
  btn.disabled = true; btn.textContent = "Detecting…";
  const isAuto = document.getElementById("auto-chk").checked;
  const thr    = isAuto ? null : parseInt(document.getElementById("thr-slider").value);
  try {
    const res  = await fetch("/api/detect", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ threshold: thr, boundary: boundary || null })
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

    slots = data.slots.map((pts, i) => ({ id: nextId++, name: "tool_" + (i + 1), points: pts }));
    pick(null); renderList(); redraw();
    showPage("edit");
    toast(`Detected ${slots.length} slot${slots.length !== 1 ? "s" : ""}`);
  } catch (err) {
    alert("Error: " + err);
  } finally {
    btn.disabled = false; btn.textContent = "Detect Slots";
  }
}

function rescan() {
  if (slots.length > 0 && !confirm("Re-scan will replace current slots. Continue?")) return;
  showPage("scan");
}

// ── save ──────────────────────────────────────────────────────────────────────
async function saveConfig() {
  const btn = document.getElementById("btn-save");
  btn.disabled = true; btn.textContent = "Saving…";
  const payload = {
    slots:    slots.map(s => ({ name: s.name || ("tool_" + s.id), points: s.points })),
    boundary: boundary || null
  };
  try {
    const res  = await fetch("/api/save", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) { alert("Save failed: " + (data.error || "unknown")); return; }
    document.getElementById("done-count").textContent = data.count;
    showPage("done");
  } finally {
    btn.disabled = false; btn.textContent = "✓ Save Configuration";
  }
}
</script>
</body>
</html>"""


# ── HTTP handler ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

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
            with calib["lock"]: jpg = calib["capture_jpg"]
            if jpg:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
            else:
                self.send_response(404); self.end_headers()
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        cl   = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(cl) if cl else b""

        if self.path == "/api/detect":
            data  = json.loads(body) if body else {}
            thr   = data.get("threshold")
            with raw_frame_lock: frame = raw_frame
            if frame is None:
                self.send_json(500, {"error": "No camera frame yet"}); return
            cv2.imwrite(BASELINE_FILE, frame)
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with calib["lock"]:
                calib["capture_jpg"] = jpg.tobytes()
                calib["frame_size"]  = (frame.shape[1], frame.shape[0])
            bdry     = data.get("boundary") or []
            detected = detect_foam_slots(frame, thr, boundary_pts=bdry or None)
            self.send_json(200, {"slots": detected, "count": len(detected)})

        elif self.path == "/api/split_join":
            data     = json.loads(body)
            slots_in = data.get("slots", [])
            line     = data.get("line", [])
            if len(line) != 2:
                self.send_json(400, {"error": "Need exactly 2 line points"}); return
            with calib["lock"]:
                fw, fh = calib.get("frame_size", (1280, 720))
            msg, removed, added = process_line(line[0], line[1], slots_in, fw, fh)
            if not removed:
                self.send_json(200, {"action": "none", "message": msg})
            else:
                self.send_json(200, {
                    "action":      "split" if len(removed) == 1 else "join",
                    "message":     msg,
                    "removed_ids": removed,
                    "added":       added,
                })

        elif self.path == "/api/save":
            data     = json.loads(body)
            slots_in = data.get("slots", [])
            bdry_in  = data.get("boundary") or []
            if not slots_in:
                self.send_json(400, {"error": "No slots defined"}); return
            slots_out = []
            for i, s in enumerate(slots_in):
                pts = [[int(p[0]), int(p[1])] for p in s["points"]]
                xs  = [p[0] for p in pts]; ys = [p[1] for p in pts]
                slots_out.append({
                    "name":                (s.get("name") or f"tool_{i+1}").strip(),
                    "x":                   min(xs), "y": min(ys),
                    "w":                   max(xs) - min(xs), "h": max(ys) - min(ys),
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
            if bdry_in:
                config["foam_boundary"] = [[int(p[0]), int(p[1])] for p in bdry_in]
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
            with calib["lock"]:
                calib["phase"] = "done"
            self.send_json(200, {"ok": True, "count": len(slots_out)})
            stop_event.set()

        else:
            self.send_response(404); self.end_headers()


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
            print(f"[calibrate] Saved to {CONFIG_FILE}.")
        else:
            print("[calibrate] Cancelled — nothing saved.")


if __name__ == "__main__":
    main()
