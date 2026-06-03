#!/usr/bin/env python3
import cv2
import json
import numpy as np
import os
import sys
from datetime import datetime

CONFIG_FILE = "toolchest_config.json"
BASELINE_FILE = "baseline.png"

def open_camera():
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_still_configuration(main={"size": (1280, 720), "format": "RGB888"}))
        cam.start()
        import time; time.sleep(2)
        print("[camera] Using PiCamera2")
        return cam, "picamera2"
    except Exception:
        pass

    for index in range(4):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print(f"[camera] Using OpenCV VideoCapture (index {index})")
            return cap, "opencv"

    print("[ERROR] No camera found. Connect a camera and retry.")
    sys.exit(1)

def capture_frame(cam, mode):
    if mode == "picamera2":
        return cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
    else:
        for _ in range(5): cam.read()
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to grab frame from camera.")
            sys.exit(1)
        return frame

def release_camera(cam, mode):
    if mode == "picamera2": cam.stop()
    else: cam.release()

current_pts = []
preview_frame = None
display_frame = None

def mouse_callback(event, x, y, flags, param):
    global current_pts, display_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        current_pts.append((x, y))
        display_frame = preview_frame.copy()
        if len(current_pts) > 1:
            for i in range(len(current_pts) - 1):
                cv2.line(display_frame, current_pts[i], current_pts[i+1], (255, 0, 0), 2)
        for pt in current_pts:
            cv2.circle(display_frame, pt, 4, (0, 0, 255), -1)
        cv2.imshow("Calibration Window", display_frame)

def get_on_screen_name(base_frame, slot_idx):
    """Draws a custom visual text box directly on screen to collect tool names cleanly."""
    input_str = ""
    while True:
        dialog = base_frame.copy()
        h, w = dialog.shape[:2]
        
        # Draw a translucent layout box on top of the screen
        cv2.rectangle(dialog, (int(w/2) - 250, int(h/2) - 60), (int(w/2) + 250, int(h/2) + 40), (30, 41, 59), -1)
        cv2.rectangle(dialog, (int(w/2) - 250, int(h/2) - 60), (int(w/2) + 250, int(h/2) + 40), (71, 85, 105), 2)
        
        prompt = f"Name for Tool #{slot_idx} (Press Enter to Save):"
        cv2.putText(dialog, prompt, (int(w/2) - 230, int(h/2) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (226, 232, 240), 1, cv2.LINE_AA)
        
        # Render what you type live
        display_text = input_str + "_"
        cv2.rectangle(dialog, (int(w/2) - 230, int(h/2) - 5), (int(w/2) + 230, int(h/2) + 25), (15, 17, 21), -1)
        cv2.putText(dialog, display_text, (int(w/2) - 220, int(h/2) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (59, 130, 246), 2, cv2.LINE_AA)
        
        cv2.imshow("Calibration Window", dialog)
        key = cv2.waitKey(0)
        
        if key == 13: # ENTER Key
            final_name = input_str.strip()
            return final_name if final_name else f"tool_{slot_idx}"
        elif key == 8 or key == 127: # Backspace Key
            input_str = input_str[:-1]
        elif 32 <= key <= 126: # Regular characters
            input_str += chr(key)

def define_slots_polygon(frame):
    global preview_frame, display_frame, current_pts
    slots = []
    
    cv2.namedWindow("Calibration Window")
    cv2.setMouseCallback("Calibration Window", mouse_callback)

    while True:
        preview_frame = frame.copy()
        # Draw saved items
        for slot in slots:
            pts_arr = np.array(slot["polygon_pts"], np.int32).reshape((-1, 1, 2))
            cv2.polylines(preview_frame, [pts_arr], True, (0, 210, 90), 2)
            cv2.putText(preview_frame, slot["name"], (slot["x"], slot["y"] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 90), 1)
        
        display_frame = preview_frame.copy()
        current_pts = []
        
        # Add visual instructions along the header banner area
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 40), (15, 17, 21), -1)
        instructions = "Left-Click to draw lines | Press ENTER to complete a shape | Press ESC when all tools are completed"
        cv2.putText(display_frame, instructions, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (241, 245, 249), 1, cv2.LINE_AA)
        
        cv2.imshow("Calibration Window", display_frame)
        
        slot_finished = False
        while not slot_finished:
            key = cv2.waitKey(20) & 0xFF
            if key == 13: # ENTER
                if len(current_pts) >= 3:
                    # Dynamically connect the last line segment back to point 1 visually
                    cv2.line(display_frame, current_pts[-1], current_pts[0], (255, 0, 0), 2)
                    
                    # Pop open our custom screen dialog box to name the tool safely
                    name = get_on_screen_name(display_frame, len(slots) + 1)
                    
                    xs = [p[0] for p in current_pts]
                    ys = [p[1] for p in current_pts]
                    x, y, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
                    
                    slots.append({
                        "name": name,
                        "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                        "polygon_pts": current_pts.copy(),
                        "baseline_dark_ratio": 0.0
                    })
                    slot_finished = True
                else:
                    print("[Warning] Click at least 3 points to create a enclosed box.")
            elif key == ord('c') or key == ord('C'):
                current_pts = []
                display_frame = preview_frame.copy()
                cv2.imshow("Calibration Window", display_frame)
            elif key == 27: # ESC
                cv2.destroyAllWindows()
                return slots

def main():
    print("=" * 55)
    print("        Smart Tool Chest Calibration Tool")
    print("=" * 55)
    
    print("\n⚠️ DRAWER ENVIRONMENT VERIFICATION:")
    print("Please completely clear out your drawer. There must be NO TOOLS present.")
    confirm = input("Are all tools removed? (type 'yes' to proceed): ").strip().lower()
    
    if confirm != 'yes':
        print("[Cancelled] Calibration aborted.")
        sys.exit(0)
        
    print("\n[Proceeding] Starting camera canvas engine...")
    cam, mode = open_camera()
    
    try:
        frame = capture_frame(cam, mode)
        cv2.imwrite(BASELINE_FILE, frame)
        print(f"[Success] Base empty file frame mapped to {BASELINE_FILE}")
        
        slots = define_slots_polygon(frame)
        if not slots:
            print("[Exit] No configurations committed.")
            return
            
        config = {
            "created": datetime.now().isoformat(),
            "baseline_image": BASELINE_FILE,
            "slots": slots,
            "settings": {
                "detection_mode": "background_subtract",
                "bg_diff_threshold": 30,
                "present_pixel_ratio": 0.05
            }
        }
        
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
            
        print(f"\n[OK] Configuration complete! {len(slots)} custom profiles bound to {CONFIG_FILE}")
        
    finally:
        release_camera(cam, mode)

if __name__ == "__main__":
    main()
