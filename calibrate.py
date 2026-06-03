#!/usr/bin/env python3
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

calib_state = {
    "phase":        "confirm",  # confirm → drawing → done
    "slots":        [],
    "baseline_jpg": None,
    "lock":         threading.Lock(),
}

cam          = None
cam_mode     = None
raw_frame    = None
raw_frame_lock = threading.Lock()
jpg_lock     = threading.Lock()
jpg_buffer   = b""
stop_event   = threading.Event()

def open_camera():
    try:
        from picamera2 import Picamera2
        c = Picamera2()
        c.configure(c.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"}))
        c.start()
        time.sleep(1)
        print("[camera] Using PiCamera2")
        return c, "picamera2"
    except Exception:
        pass
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[camera] Using OpenCV VideoCapture (index {idx})")
            return cap, "opencv"
    print("[ERROR] No camera found.")
    sys.exit(1)

def read_frame(c, mode):
    if mode == "picamera2":
        return cv2.cvtColor(c.capture_array(), cv2.COLOR_RGB2BGR)
    c.grab()
    ret, frame = c.retrieve()
    if not ret:
        raise RuntimeError("Camera read failed")
    return frame

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
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                    str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
                )
            time.sleep(0.04)
    except Exception:
        pass

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Tool Chest Calibration</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: sans-serif; background: #0f1115; color: #e2e8f0; padding: 24px; margin: 0; }
  h1 { color: #3b82f6; margin-top: 0; }
  .step { display: none; }
  .step.active { display: block; }
  .instructions { background: #1e2330; border: 1px solid #374151; border-radius: 8px; padding: 14px 18px; margin-bottom: 18px; color: #94a3b8; line-height: 1.6; }
  .instructions b { color: #e2e8f0; }
  .btn { background: #3b82f6; color: #fff; border: none; padding: 9px 20px; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 600; transition: background 0.15s; }
  .btn:hover { background: #2563eb; }
  .btn:disabled { background: #374151; color: #6b7280; cursor: default; }
  .btn-danger  { background: #dc2626; } .btn-danger:hover  { background: #b91c1c; }
  .btn-success { background: #16a34a; } .btn-success:hover { background: #15803d; }
  .btn-muted   { background: #374151; } .btn-muted:hover   { background: #4b5563; }
  .controls { margin: 14px 0; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  #live-feed { max-width: 800px; width: 100%; border: 2px solid #374151; border-radius: 8px; display: block; }
  #draw-canvas { max-width: 800px; width: 100%; cursor: crosshair; border: 2px solid #3b82f6; border-radius: 8px; display: block; }
  #slot-name { background: #1e2330; color: #e2e8f0; border: 1px solid #374151; border-radius: 6px; padding: 8px 12px; font-size: 14px; width: 220px; }
  #slot-name:focus { outline: none; border-color: #3b82f6; }
  .slot-list { margin-top: 12px; }
  .slot-item { background: #1e2330; padding: 7px 14px; border-radius: 6px; margin-bottom: 6px; border-left: 3px solid #16a34a; color: #a3e635; font-size: 14px; display: flex; justify-content: space-between; align-items: center; }
  .slot-item span { color: #94a3b8; font-size: 12px; }
  .pt-count { color: #94a3b8; font-size: 13px; }
  .done-msg { color: #4ade80; font-size: 1.3em; margin-bottom: 10px; }
  code { background: #1e2330; padding: 2px 6px; border-radius: 4px; color: #93c5fd; }
</style>
</head>
<body>
<h1>&#129520; Tool Chest &mdash; Web Calibration</h1>

<!-- Step 1: confirm empty drawer, capture baseline -->
<div id="step-confirm" class="step active">
  <div class="instructions">
    <b>Step 1 &mdash; Capture Baseline</b><br>
    Make sure the drawer is <b>completely empty</b> (no tools present).<br>
    Point the camera at the empty drawer, then click <b>Capture Baseline</b>.
  </div>
  <img id="live-feed" src="/stream">
  <div class="controls">
    <button class="btn" onclick="captureBaseline()">Capture Baseline</button>
  </div>
</div>

<!-- Step 2: draw polygons -->
<div id="step-draw" class="step">
  <div class="instructions">
    <b>Step 2 &mdash; Define Tool Slots</b><br>
    Click on the image to place points around a tool slot (at least 3 points).<br>
    Click <b>Finish Slot</b>, enter a name, then <b>Add Slot</b>. Repeat for every slot.<br>
    Press <b>Save Configuration</b> when all slots are defined.
  </div>
  <canvas id="draw-canvas"></canvas>
  <div class="controls">
    <button class="btn btn-muted" onclick="undoPoint()">&#8592; Undo Point</button>
    <button class="btn" id="btn-finish" onclick="finishSlot()" disabled>Finish Slot</button>
    <span class="pt-count" id="pt-count">0 points</span>
  </div>
  <div class="controls" id="name-row" style="display:none;">
    <input type="text" id="slot-name" placeholder="Tool name (e.g. pliers)&hellip;"
           onkeydown="if(event.key==='Enter') addSlot()">
    <button class="btn btn-success" onclick="addSlot()">Add Slot</button>
    <button class="btn btn-danger"  onclick="cancelSlot()">Cancel</button>
  </div>
  <div class="slot-list" id="slot-list"></div>
  <div class="controls" style="margin-top: 20px;">
    <button class="btn btn-success" id="btn-save" onclick="saveConfig()" disabled>
      &#10003; Save Configuration
    </button>
  </div>
</div>

<!-- Step 3: done -->
<div id="step-done" class="step">
  <p class="done-msg">&#10003; Calibration complete!</p>
  <p style="color:#94a3b8;">Configuration saved to <code>toolchest_config.json</code>.<br>
  You can close this tab and start <code>detect.py</code>.</p>
</div>

<script>
const canvas = document.getElementById("draw-canvas");
const ctx    = canvas.getContext("2d");
let currentPts    = [];
let completedSlots = [];
let baselineImg   = null;
let drawingActive = true;
let pendingPts    = null;

function showStep(name) {
  document.querySelectorAll(".step").forEach(s => s.classList.remove("active"));
  document.getElementById("step-" + name).classList.add("active");
}

async function captureBaseline() {
  const res = await fetch("/api/capture", { method: "POST" });
  if (!res.ok) { alert("Capture failed — is the camera connected?"); return; }
  baselineImg = new Image();
  baselineImg.onload = () => {
    canvas.width  = baselineImg.naturalWidth;
    canvas.height = baselineImg.naturalHeight;
    redraw();
    showStep("draw");
  };
  baselineImg.src = "/baseline.jpg?" + Date.now();
}

canvas.addEventListener("click", e => {
  if (!drawingActive) return;
  const r  = canvas.getBoundingClientRect();
  const sx = canvas.width  / r.width;
  const sy = canvas.height / r.height;
  const x  = Math.round((e.clientX - r.left) * sx);
  const y  = Math.round((e.clientY - r.top)  * sy);
  currentPts.push([x, y]);
  document.getElementById("pt-count").textContent = currentPts.length + " point" + (currentPts.length !== 1 ? "s" : "");
  document.getElementById("btn-finish").disabled = currentPts.length < 3;
  redraw();
});

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (baselineImg) ctx.drawImage(baselineImg, 0, 0);

  // Completed slots
  completedSlots.forEach(s => {
    ctx.beginPath();
    s.points.forEach((p, i) => i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]));
    ctx.closePath();
    ctx.strokeStyle = "#00d262";
    ctx.lineWidth   = 2;
    ctx.setLineDash([]);
    ctx.stroke();
    ctx.fillStyle = "rgba(0,210,98,0.15)";
    ctx.fill();
    ctx.fillStyle  = "#00d262";
    ctx.font       = "bold 15px sans-serif";
    ctx.shadowColor = "#000"; ctx.shadowBlur = 4;
    ctx.fillText(s.name, s.points[0][0] + 5, s.points[0][1] - 7);
    ctx.shadowBlur = 0;
  });

  // In-progress polygon
  if (currentPts.length > 0) {
    ctx.beginPath();
    currentPts.forEach((p, i) => i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]));
    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth   = 2;
    ctx.setLineDash([7, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
    currentPts.forEach(p => {
      ctx.beginPath();
      ctx.arc(p[0], p[1], 5, 0, 2 * Math.PI);
      ctx.fillStyle = "#ef4444";
      ctx.fill();
    });
  }
}

function undoPoint() {
  currentPts.pop();
  document.getElementById("pt-count").textContent = currentPts.length + " point" + (currentPts.length !== 1 ? "s" : "");
  document.getElementById("btn-finish").disabled = currentPts.length < 3;
  redraw();
}

function finishSlot() {
  if (currentPts.length < 3) return;
  drawingActive = false;
  pendingPts    = [...currentPts];
  document.getElementById("name-row").style.display = "flex";
  document.getElementById("slot-name").value = "";
  document.getElementById("slot-name").focus();
}

function cancelSlot() {
  drawingActive = true;
  pendingPts    = null;
  document.getElementById("name-row").style.display = "none";
}

async function addSlot() {
  const raw  = document.getElementById("slot-name").value.trim();
  const name = raw || ("tool_" + (completedSlots.length + 1));
  const pts  = pendingPts;
  const res  = await fetch("/api/add_slot", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, points: pts })
  });
  if (!res.ok) { alert("Failed to add slot"); return; }

  completedSlots.push({ name, points: pts });
  currentPts    = [];
  pendingPts    = null;
  drawingActive = true;
  document.getElementById("name-row").style.display  = "none";
  document.getElementById("pt-count").textContent    = "0 points";
  document.getElementById("btn-finish").disabled     = true;
  document.getElementById("btn-save").disabled       = false;

  const listEl = document.getElementById("slot-list");
  listEl.innerHTML = completedSlots.map((s, i) =>
    `<div class="slot-item">#${i + 1}: ${s.name} <span>${s.points.length} pts</span></div>`
  ).join("");
  redraw();
}

async function saveConfig() {
  if (completedSlots.length === 0) { alert("Define at least one slot first."); return; }
  const res = await fetch("/api/save", { method: "POST" });
  if (!res.ok) { alert("Save failed"); return; }
  showStep("done");
}
</script>
</body>
</html>"""


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

        elif self.path.startswith("/baseline.jpg"):
            with calib_state["lock"]:
                jpg = calib_state["baseline_jpg"]
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

        if self.path == "/api/capture":
            with raw_frame_lock:
                frame = raw_frame
            if frame is None:
                self.send_json(500, {"error": "No camera frame yet"}); return
            cv2.imwrite(BASELINE_FILE, frame)
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with calib_state["lock"]:
                calib_state["baseline_jpg"] = jpg.tobytes()
                calib_state["phase"]        = "drawing"
            self.send_json(200, {"ok": True})

        elif self.path == "/api/add_slot":
            data = json.loads(body)
            pts  = [[int(p[0]), int(p[1])] for p in data["points"]]
            xs   = [p[0] for p in pts]
            ys   = [p[1] for p in pts]
            slot = {
                "name":                data["name"].strip() or f"tool_{len(calib_state['slots']) + 1}",
                "x":                   min(xs),
                "y":                   min(ys),
                "w":                   max(xs) - min(xs),
                "h":                   max(ys) - min(ys),
                "polygon_pts":         pts,
                "baseline_dark_ratio": 0.0,
            }
            with calib_state["lock"]:
                calib_state["slots"].append(slot)
            self.send_json(200, {"ok": True, "slot_count": len(calib_state["slots"])})

        elif self.path == "/api/save":
            with calib_state["lock"]:
                slots = list(calib_state["slots"])
            if not slots:
                self.send_json(400, {"error": "No slots defined"}); return
            config = {
                "created":        datetime.now().isoformat(),
                "baseline_image": BASELINE_FILE,
                "slots":          slots,
                "settings": {
                    "detection_mode":      "background_subtract",
                    "bg_diff_threshold":   30,
                    "present_pixel_ratio": 0.05,
                },
            }
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
            with calib_state["lock"]:
                calib_state["phase"] = "done"
            self.send_json(200, {"ok": True})
            stop_event.set()

        else:
            self.send_response(404)
            self.end_headers()


class ThreadedHTTP(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    global cam, cam_mode
    cam, cam_mode = open_camera()
    threading.Thread(target=capture_loop, daemon=True).start()

    server = ThreadedHTTP(("0.0.0.0", PORT), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print(f"[calibrate] Open http://localhost:{PORT} in your browser to begin.")

    try:
        stop_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        release_camera(cam, cam_mode)
        if calib_state["phase"] == "done":
            print(f"[calibrate] Configuration saved to {CONFIG_FILE}")
        else:
            print("[calibrate] Cancelled — no configuration saved.")


if __name__ == "__main__":
    main()
