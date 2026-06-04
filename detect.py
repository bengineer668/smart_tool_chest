#!/usr/bin/env python3
import argparse
import cv2
import json
import numpy as np
import os
import sys
import time
import threading
from datetime import datetime

# Initialize Pygame Audio Mixer safely using system default audio paths
try:
    import pygame
    pygame.mixer.init()
    AUDIO_ENABLED = True
    print("[audio] Pygame mixer initialized successfully using system defaults.")
except Exception as e:
    AUDIO_ENABLED = False
    print(f"[audio warning] Could not initialize audio player: {e}")

CONFIG_FILE  = "toolchest_config.json"
LOG_FILE     = "tool_events.jsonl"

raw_frame_lock = threading.Lock()
raw_frame      = None

jpg_lock   = threading.Lock()
jpg_buffer = b""

state = {
    "slots":         {},
    "last_frame_ts": None,
    "missing_tools": [],
    "locked":        True,
    "current_user":  "None (Locked)",
    "lock":          threading.Lock()
}

def play_sound(sound_type):
    """Explicitly tells Pygame to play a specific sound file at maximum volume."""
    if not AUDIO_ENABLED:
        return
    def _play():
        try:
            sound_path = f"sounds/{sound_type}.wav"
            if os.path.exists(sound_path):
                sound = pygame.mixer.Sound(sound_path)
                sound.set_volume(1.0) # Force maximum sound volume output (1.0 = 100%)
                sound.play()
            else:
                print(f"[audio missing] Sound file not found: {sound_path}")
        except Exception as e:
            print(f"[audio error] Failed to play {sound_type}: {e}")
    
    # Run in background thread so the camera stream never stutters
    threading.Thread(target=_play, daemon=True).start()

def open_camera():
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"}))
        cam.start()
        time.sleep(1)
        print("[camera] Using PiCamera2")
        return cam, "picamera2"
    except Exception:
        pass

    for index in range(4):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[camera] Using OpenCV VideoCapture (index {index})")
            return cap, "opencv"
    print("[ERROR] No camera found.")
    sys.exit(1)

def read_frame(cam, mode):
    if mode == "picamera2":
        return cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
    else:
        cam.grab()
        ret, frame = cam.retrieve()
        if not ret: raise RuntimeError("Camera read failed")
        return frame

def release_camera(cam, mode):
    if mode == "picamera2": cam.stop()
    else: cam.release()

def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"[ERROR] {CONFIG_FILE} not found. Run calibrate.py first.")
        sys.exit(1)
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    cfg["settings"].setdefault("detection_mode", "background_subtract")
    cfg["settings"].setdefault("bg_diff_threshold", 30)
    cfg["settings"].setdefault("present_pixel_ratio", 0.05)
    return cfg

def update_config_tool_user(slot_name, current_holder, last_history, timestamp):
    try:
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
        for slot in cfg["slots"]:
            if slot["name"] == slot_name:
                slot["current_holder"] = current_holder
                slot["last_user"] = last_history
                slot["last_event_ts"] = timestamp
                break
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"[config error] {e}")

def load_baseline_image(config):
    path = config.get("baseline_image", "baseline.png")
    img  = cv2.imread(path)
    if img is None:
        print(f"[ERROR] Baseline image not found at {path}. Re-run calibrate.py.")
        sys.exit(1)
    return img

def detect_tools(frame_bgr, baseline_bgr, config):
    s = config["settings"]
    if s.get("detection_mode", "background_subtract") == "background_subtract":
        diff_thresh = int(s.get("bg_diff_threshold", 30))
        present_ratio = float(s.get("present_pixel_ratio", 0.05))
        if baseline_bgr.shape != frame_bgr.shape:
            baseline_bgr = cv2.resize(baseline_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]))
        gray_cur  = cv2.GaussianBlur(cv2.cvtColor(frame_bgr,    cv2.COLOR_BGR2GRAY), (5, 5), 0)
        gray_base = cv2.GaussianBlur(cv2.cvtColor(baseline_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        diff = cv2.absdiff(gray_cur, gray_base)
        _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        if config.get("foam_boundary"):
            bpts  = np.array(config["foam_boundary"], np.int32)
            bmask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillPoly(bmask, [bpts], 255)
            mask  = cv2.bitwise_and(mask, bmask)
        results = {}
        for slot in config["slots"]:
            if "polygon_pts" in slot:
                pts = np.array(slot["polygon_pts"], np.int32)
                poly_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.fillPoly(poly_mask, [pts], 255)
                roi = cv2.bitwise_and(mask, poly_mask)
                w, h = slot["w"], slot["h"]
                changed_ratio = float(np.sum(roi > 0)) / (w * h)
            else:
                x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
                roi = mask[y:y+h, x:x+w]
                changed_ratio = float(np.sum(roi > 0)) / (w * h)
            results[slot["name"]] = {"present": changed_ratio >= present_ratio, "dark_ratio": round(changed_ratio, 4)}
        return results
    return {}

def annotate_and_encode(frame_bgr, config):
    vis = frame_bgr.copy()
    with state["lock"]:
        slots_snapshot = dict(state["slots"])
        is_locked = state["locked"]
    for slot in config["slots"]:
        name = slot["name"]
        present = slots_snapshot.get(name, {}).get("present", True)
        color = (0, 210, 90) if present else (30, 30, 220)
        if "polygon_pts" in slot:
            pts = np.array(slot["polygon_pts"], np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, color, 2)
            cv2.putText(vis, f"{'OK' if present else 'OUT'} {name}", (slot["x"] + 4, slot["y"] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis, f"{'OK' if present else 'OUT'} {name}", (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if config.get("foam_boundary"):
        bpts = np.array(config["foam_boundary"], np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [bpts], True, (100, 200, 255), 2)
    lock_str, lock_col = ("LOCKED", (0, 0, 200)) if is_locked else ("UNLOCKED", (0, 180, 0))
    cv2.rectangle(vis, (10, 10), (160, 45), lock_col, -1)
    cv2.putText(vis, lock_str, (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    h_img, w_img = vis.shape[:2]
    scale = min(800 / w_img, 1.0)
    if scale < 1.0: vis = cv2.resize(vis, (int(w_img * scale), int(h_img * scale)))
    _, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return jpg.tobytes()

def capture_loop(cam, mode, config):
    global raw_frame, jpg_buffer
    while True:
        try:
            frame = read_frame(cam, mode)
            with raw_frame_lock: raw_frame = frame
            jpg = annotate_and_encode(frame, config)
            with jpg_lock: jpg_buffer = jpg
        except Exception: time.sleep(0.1)

def log_event(slot_name, event_str, target_user, dark_ratio):
    event = {"ts": datetime.now().isoformat(), "slot": slot_name, "event": event_str, "user": target_user, "dark_ratio": dark_ratio}
    with open(LOG_FILE, "a") as f: f.write(json.dumps(event) + "\n")

def detection_loop(config, baseline_bgr, interval=0.5):
    prev_detection = {}
    with state["lock"]:
        for slot in config["slots"]:
            state["slots"][slot["name"]] = {
                "present": True, "dark_ratio": 0.0,
                "current_holder": slot.get("current_holder", "—"),
                "last_user": slot.get("last_user", "—"),
                "last_event_ts": slot.get("last_event_ts", "—")
            }
    while True:
        with raw_frame_lock: frame = raw_frame
        if frame is None:
            time.sleep(0.1)
            continue
        try:
            detection = detect_tools(frame, baseline_bgr, config)
            ts_readable = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with state["lock"]:
                is_locked = state["locked"]
                active_user = state["current_user"]

            if not is_locked:
                for name, info in detection.items():
                    prev = prev_detection.get(name, {})
                    if prev and prev.get("present") != info["present"]:
                        # 1. TOOL REMOVED -> Low "Boopy" Sound
                        if not info["present"]:
                            play_sound("missing_boop")
                            log_event(name, "removed", active_user, info["dark_ratio"])
                            with state["lock"]:
                                state["slots"][name]["current_holder"] = active_user
                                state["slots"][name]["last_event_ts"] = ts_readable
                            update_config_tool_user(name, active_user, state["slots"][name]["last_user"], ts_readable)
                        
                        # 2. TOOL RETURNED -> Higher Pitch Confirmation Sound
                        elif info["present"]:
                            play_sound("return_high")
                            log_event(name, "returned", active_user, info["dark_ratio"])
                            with state["lock"]:
                                taker = state["slots"][name]["current_holder"]
                                if taker == "—" or not taker: taker = "Unknown"
                                if taker == active_user:
                                    history_str = active_user
                                else:
                                    history_str = f"O: {taker} / I: {active_user}"
                                state["slots"][name]["current_holder"] = "—"
                                state["slots"][name]["last_user"] = history_str
                                state["slots"][name]["last_event_ts"] = ts_readable
                            update_config_tool_user(name, "—", history_str, ts_readable)

            with state["lock"]:
                for name, info in detection.items():
                    if name in state["slots"]:
                        state["slots"][name]["present"] = info["present"]
                        state["slots"][name]["dark_ratio"] = info["dark_ratio"]
                state["last_frame_ts"] = datetime.now().isoformat()
                state["missing_tools"] = [n for n, i in detection.items() if not i["present"]]
            if not is_locked or not prev_detection: prev_detection = detection
        except Exception: pass
        time.sleep(interval)

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Team 668 &mdash; Smart Tool Chest</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #000; color: #d0d0d0; min-height: 100vh; }
  header {
    background: #0a0000;
    border-bottom: 2px solid #a80000;
    padding: 14px 24px;
    display: flex; align-items: baseline; gap: 16px;
  }
  header h1 { font-size: 1.25rem; font-weight: 700; color: #a80000; letter-spacing: 0.04em; }
  header span { font-size: 0.8rem; color: #666; }
  .layout { display: flex; gap: 16px; padding: 16px; max-width: 1400px; margin: 0 auto; }
  .live-wrap { flex: 1 1 60%; background: #000; border: 1px solid #2a0000; border-radius: 8px; overflow: hidden; }
  .live-wrap img { width: 100%; display: block; }
  .sidebar { flex: 0 0 38%; display: flex; flex-direction: column; gap: 14px; }
  .panel { background: #0d0d0d; border: 1px solid #2a0000; border-radius: 8px; padding: 15px; }
  .panel h3 { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #555; margin-bottom: 12px; }
  .status-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
  .status-label { font-size: 0.9rem; color: #888; }
  .badge { padding: 3px 9px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em; }
  .badge.locked   { background: #3a0000; color: #cc6666; border: 1px solid #600000; }
  .badge.unlocked { background: #0a2a0a; color: #66cc66; border: 1px solid #1a5a1a; }
  .badge.in  { background: #0a2a0a; color: #66cc66; border: 1px solid #1a5a1a; }
  .badge.out { background: #3a0000; color: #cc6666; border: 1px solid #600000; }
  .user-line { font-size: 0.88rem; color: #777; margin-bottom: 12px; }
  .user-line b { color: #a80000; font-weight: 600; }
  .auth-row { display: flex; gap: 8px; }
  input[type=text] {
    background: #0a0000; border: 1px solid #3a0000; color: #d0d0d0;
    border-radius: 5px; padding: 7px 10px; font-size: 0.88rem; flex: 1;
  }
  input[type=text]:focus { outline: none; border-color: #a80000; }
  button {
    background: #a80000; color: #fff; border: none;
    padding: 7px 14px; border-radius: 5px; font-size: 0.85rem;
    font-weight: 600; cursor: pointer; transition: background 0.15s;
  }
  button:hover { background: #cc0000; }
  .btn-lock { background: #3a0000; color: #cc6666; width: 100%; padding: 9px; }
  .btn-lock:hover { background: #600000; color: #ff9999; }
  table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
  thead th {
    text-align: left; padding: 5px 8px; color: #555;
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
    border-bottom: 1px solid #1a0000;
  }
  tbody td { padding: 8px; border-bottom: 1px solid #161616; vertical-align: middle; }
  tbody tr:hover td { background: #0f0000; }
  .tool-name { font-weight: 600; color: #c8c8c8; }
  .holder { color: #a80000; font-weight: 600; }
  .ts { color: #444; font-size: 0.77rem; }
</style>
<script>
  let lastLockState = null;
  async function refresh() {
    try {
      const r = await fetch('/api/status');
      const d = await r.json();
      const badge = document.getElementById('lock-badge');
      badge.className = 'badge ' + (d.locked ? 'locked' : 'unlocked');
      badge.textContent = d.locked ? 'LOCKED' : 'UNLOCKED';
      document.getElementById('auth-user').textContent = d.current_user || '—';
      if (lastLockState !== d.locked) {
        lastLockState = d.locked;
        document.getElementById('action-area').innerHTML = d.locked
          ? `<div class="auth-row"><input type="text" id="username" placeholder="Enter your name to unlock..."><button onclick="auth()">Unlock</button></div>`
          : `<button class="btn-lock" onclick="lock()">Lock Drawer</button>`;
      }
      let rows = '';
      for (const [name, info] of Object.entries(d.slots)) {
        rows += `<tr>
          <td class="tool-name">${name}</td>
          <td><span class="badge ${info.present ? 'in' : 'out'}">${info.present ? 'PRESENT' : 'MISSING'}</span></td>
          <td class="holder">${info.current_holder || '&mdash;'}</td>
          <td>${info.last_user || '&mdash;'}</td>
          <td class="ts">${info.last_event_ts || '&mdash;'}</td>
        </tr>`;
      }
      document.getElementById('inventory-body').innerHTML = rows;
    } catch(e) {}
  }
  async function auth() {
    const u = document.getElementById('username').value.trim();
    if (!u) return;
    await fetch('/api/unlock', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:'user='+encodeURIComponent(u)});
    refresh();
  }
  async function lock() { await fetch('/api/lock', {method:'POST'}); refresh(); }
  window.addEventListener('load', () => { document.getElementById('live-feed').src = '/stream'; setInterval(refresh, 1000); });
</script>
</head>
<body>
  <header>
    <h1>Smart Tool Chest</h1>
    <span>Team 668 &mdash; Apes of Wrath</span>
  </header>
  <div class="layout">
    <div class="live-wrap"><img id="live-feed" src=""></div>
    <div class="sidebar">
      <div class="panel">
        <h3>Status</h3>
        <div class="status-row">
          <span class="status-label">Drawer</span>
          <span id="lock-badge" class="badge locked">LOCKED</span>
        </div>
        <div class="user-line">User: <b id="auth-user">None</b></div>
        <div id="action-area"></div>
      </div>
      <div class="panel">
        <h3>Tool Inventory</h3>
        <table>
          <thead><tr><th>Tool</th><th>Status</th><th>Taken by</th><th>Last user</th><th>Timestamp</th></tr></thead>
          <tbody id="inventory-body"></tbody>
        </table>
      </div>
    </div>
  </div>
</body>
</html>"""

def mjpeg_stream(handler):
    handler.send_response(200)
    handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
    handler.end_headers()
    try:
        while True:
            with jpg_lock: jpg = jpg_buffer
            if jpg: handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.04)
    except Exception: pass

def start_web_server(port):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from socketserver import ThreadingMixIn
    from urllib.parse import parse_qs
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args): pass
        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(HTML_TEMPLATE.encode())
            elif self.path == "/stream": mjpeg_stream(self)
            elif self.path == "/api/status":
                with state["lock"]: p = json.dumps({"slots": state["slots"], "locked": state["locked"], "current_user": state["current_user"]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(p)
        def do_POST(self):
            if self.path == "/api/unlock":
                cl = int(self.headers['Content-Length'])
                u = parse_qs(self.rfile.read(cl).decode('utf-8')).get('user', ['Unknown'])[0]
                with state["lock"]:
                    state["locked"] = False
                    state["current_user"] = u
                # Unlocked -> Play Unlocked Chime
                play_sound("unlocked_chime")
            elif self.path == "/api/lock":
                with state["lock"]:
                    state["locked"] = True
                    state["current_user"] = "None (Locked)"
                # Locked -> Play Lower Lock Tone
                play_sound("locked_lower")
            self.send_response(200)
            self.end_headers()
    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer): daemon_threads = True
    print(f"[web] Dashboard ready at http://localhost:{port}")
    ThreadedHTTPServer(("0.0.0.0", port), Handler).serve_forever()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    config = load_config()
    baseline = load_baseline_image(config)
    cam, cam_mode = open_camera()
    threading.Thread(target=capture_loop, args=(cam, cam_mode, config), daemon=True).start()
    threading.Thread(target=detection_loop, args=(config, baseline), daemon=True).start()
    try:
        import screen as tft
        threading.Thread(target=tft.display_loop, args=(state,), daemon=True).start()
    except ImportError:
        pass
    try: start_web_server(args.port)
    except KeyboardInterrupt: pass
    release_camera(cam, cam_mode)

if __name__ == "__main__":
    main()
