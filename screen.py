#!/usr/bin/env python3
"""
screen.py — Live TFT display for the Smart Tool Chest.
Renders the tool / lock status to the ILI9341 framebuffer (/dev/fb0).
Called from detect.py as a daemon thread — no extra libraries needed.

Screen: 320×240 landscape (kernel fbtft driver rotates 90°)
"""
import glob
import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── display geometry ──────────────────────────────────────────────────────────
W, H      = 320, 240
PIN_LED   = 18          # GPIO for PWM backlight
BGR       = False       # set True if red and blue appear swapped
PAGE_SIZE = 5           # tool rows per page
PAGE_SECS = 4.0         # seconds before flipping to next page

# ── colour palette ────────────────────────────────────────────────────────────
ACCENT      = (168,   0,   0)   # #a80000 — team red
ACCENT_DIM  = ( 90,   0,   0)   # darker red for subtle details
BG          = (  0,   0,   0)   # black
HEADER_BG   = ( 18,   0,   0)   # near-black red tint
GREEN       = ( 80, 200,  80)   # tool present
RED         = (220,  55,  55)   # tool missing
MUTED       = (110, 110, 110)   # grey secondary text
WHITE       = (210, 210, 210)   # light grey / primary text
DIVIDER     = ( 28,  28,  28)   # dark grey divider
PILL_LOCK   = (( 80,   0,   0), (220, 120, 120))   # (fill, text)
PILL_UNLOCK = (( 10,  55,  10), (100, 210, 100))
ALERT_MISS  = (( 55,   5,   5), (220,  60,  60))
ALERT_OK    = ((  5,  45,  10), ( 80, 200,  85))
ROW_MISS_BG = ( 38,   4,   4)

# ── framebuffer ───────────────────────────────────────────────────────────────
_fb = None


def _find_fb():
    for sysfs in sorted(glob.glob("/sys/class/graphics/fb*")):
        try:
            name = open(f"{sysfs}/name").read().strip().lower()
            if "ili9341" in name:
                return f"/dev/{os.path.basename(sysfs)}"
        except OSError:
            pass
    for dev in ("/dev/fb0", "/dev/fb1"):
        if os.path.exists(dev):
            return dev
    return None


def _write(img: Image.Image):
    if _fb is None:
        return
    a = np.array(img.convert("RGB"), dtype=np.uint16)
    r, g, b = (a[:, :, 2], a[:, :, 1], a[:, :, 0]) if BGR else \
              (a[:, :, 0], a[:, :, 1], a[:, :, 2])
    try:
        with open(_fb, "wb") as f:
            f.write((((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3))
                    .astype(np.uint16).tobytes())
    except Exception:
        pass


# ── fonts  (loaded once at startup) ───────────────────────────────────────────
F = {}


def _load_fonts():
    def font(size, bold=False):
        sfx = "-Bold" if bold else ""
        for p in (
            f"/usr/share/fonts/truetype/dejavu/DejaVuSans{sfx}.ttf",
            f"/usr/share/fonts/truetype/liberation/"
            f"LiberationSans{'-Bold' if bold else '-Regular'}.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
        return ImageFont.load_default()

    F["head"]  = font(15, bold=True)   # header title
    F["label"] = font(13, bold=True)   # section labels
    F["tool"]  = font(13)              # tool names
    F["small"] = font(10)              # badges, footer


# ── drawing helpers ───────────────────────────────────────────────────────────
def _tw(draw, text, font):
    try:
        bb = draw.textbbox((0, 0), text, font=font)
        return bb[2] - bb[0]
    except AttributeError:
        return draw.textsize(text, font=font)[0]


def _cx(draw, y, text, font, fill):
    """Draw text centred horizontally."""
    draw.text(((W - _tw(draw, text, font)) // 2, y), text, fill=fill, font=font)


def _rx(draw, y, text, font, fill, margin=8):
    """Draw text right-aligned."""
    draw.text((W - _tw(draw, text, font) - margin, y), text, fill=fill, font=font)


def _pill(draw, x1, y1, x2, y2, fill, text, font, text_fill):
    """Rounded rectangle badge with centred label."""
    try:
        draw.rounded_rectangle([(x1, y1), (x2, y2)], radius=4, fill=fill)
    except AttributeError:
        draw.rectangle([(x1, y1), (x2, y2)], fill=fill)
    tw = _tw(draw, text, font)
    tx = x1 + ((x2 - x1) - tw) // 2
    ty = y1 + ((y2 - y1) - 10) // 2
    draw.text((tx, ty), text, fill=text_fill, font=font)


def _hline(draw, y, color=None):
    draw.line([(0, y), (W, y)], fill=color or DIVIDER, width=1)


# ── render one frame ──────────────────────────────────────────────────────────
def render(snap, page=0, pulse=False):
    """
    Build a 320×240 PIL image from a state snapshot.

    snap   — dict copy of detect.py `state`
    page   — which page of tools to show (0-based, PAGE_SIZE tools per page)
    pulse  — True on every other half-second tick (makes missing dot blink)
    """
    locked  = snap.get("locked", True)
    user    = snap.get("current_user", "")
    missing = snap.get("missing_tools", [])
    slots   = snap.get("slots", {})       # {name: {present, current_holder, …}}

    # Clean up user string
    if not user or user.startswith("None"):
        user_disp = "—"
    else:
        user_disp = user

    img  = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # ────────────────────────────────────────────────────────────────────────
    # 1.  HEADER BAR  (y 0–28)
    # ────────────────────────────────────────────────────────────────────────
    draw.rectangle([(0, 0), (W, 28)], fill=HEADER_BG)
    _cx(draw, 7, "Smart Tool Chest", F["head"], ACCENT)

    # Lock pill (top-right corner)
    pill_fill, pill_txt_col = PILL_LOCK if locked else PILL_UNLOCK
    pill_label = "LOCKED" if locked else "UNLOCKED"
    pill_w     = _tw(draw, pill_label, F["small"]) + 12
    _pill(draw, W - pill_w - 5, 6, W - 5, 22,
          pill_fill, pill_label, F["small"], pill_txt_col)

    _hline(draw, 29)

    # ────────────────────────────────────────────────────────────────────────
    # 2.  USER ROW  (y 30–50)
    # ────────────────────────────────────────────────────────────────────────
    draw.rectangle([(0, 30), (W, 50)], fill=DIVIDER)
    user_line = f"User:  {user_disp}"
    draw.text((10, 35), user_line, fill=WHITE, font=F["label"])

    _hline(draw, 51)

    # ────────────────────────────────────────────────────────────────────────
    # 3.  ALERT BAR  (y 52–74)
    # ────────────────────────────────────────────────────────────────────────
    n_miss = len(missing)
    alert_bg, alert_fg = ALERT_MISS if n_miss else ALERT_OK
    draw.rectangle([(0, 52), (W, 74)], fill=alert_bg)

    if n_miss:
        names_str = ", ".join(missing)
        # Truncate if the list is too wide to fit
        while names_str and _tw(draw, f"  ⚠  Missing: {names_str}", F["label"]) > W - 12:
            names_str = names_str.rsplit(", ", 1)[0] + "…"
        alert_txt = f"  ⚠  Missing: {names_str}"
    else:
        alert_txt = "  ✓  All tools present"

    draw.text((8, 57), alert_txt, fill=alert_fg, font=F["label"])
    _hline(draw, 75)

    # ────────────────────────────────────────────────────────────────────────
    # 4.  TOOL ROWS  (y 76 downwards, PAGE_SIZE rows of 24px)
    # ────────────────────────────────────────────────────────────────────────
    ROW_H    = 24
    FOOTER_H = 14
    MAX_Y    = H - FOOTER_H - 1

    slot_items  = list(slots.items())
    total_pages = max(1, (len(slot_items) + PAGE_SIZE - 1) // PAGE_SIZE)
    page        = page % total_pages
    visible     = slot_items[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]

    y = 76
    for name, info in visible:
        if y + ROW_H > MAX_Y:
            break
        present  = info.get("present", True)
        row_bg   = ROW_MISS_BG if not present else BG
        dot_col  = GREEN if present else (RED if not pulse else (120, 30, 30))
        name_col = WHITE if present else (255, 170, 170)
        stat_txt = "PRESENT" if present else "MISSING"
        stat_col = GREEN if present else RED

        # Row background
        draw.rectangle([(0, y), (W, y + ROW_H - 1)], fill=row_bg)

        # Status dot
        draw.ellipse([(9, y + 6), (20, y + 17)], fill=dot_col)

        # Tool name — full width, never truncated
        draw.text((28, y + 5), name, fill=name_col, font=F["tool"])

        # Status label right-aligned
        _rx(draw, y + 5, stat_txt, F["small"], stat_col)

        # Row divider
        _hline(draw, y + ROW_H - 1)
        y += ROW_H

    # Overflow note
    overflow = len(slot_items) - (page + 1) * PAGE_SIZE
    if overflow > 0 and y <= MAX_Y:
        draw.text((28, y + 4),
                  f"+ {overflow} more tool{'s' if overflow > 1 else ''} → next page",
                  fill=MUTED, font=F["small"])

    # No slots configured yet
    if not slot_items:
        draw.text((28, 96), "No slots configured.", fill=MUTED, font=F["tool"])
        draw.text((28, 116), "Run calibrate.py first.", fill=MUTED, font=F["small"])

    # ────────────────────────────────────────────────────────────────────────
    # 5.  FOOTER  (bottom 14px)
    # ────────────────────────────────────────────────────────────────────────
    _hline(draw, H - FOOTER_H)
    draw.rectangle([(0, H - FOOTER_H + 1), (W, H)], fill=DIVIDER)

    ts = time.strftime("%H:%M")
    draw.text((6, H - FOOTER_H + 2), ts, fill=MUTED, font=F["small"])

    n_ok = sum(1 for i in slots.values() if i.get("present", True))
    summary = f"{n_ok} / {len(slots)} present"
    draw.text((52, H - FOOTER_H + 2), summary, fill=MUTED, font=F["small"])

    if total_pages > 1:
        page_txt = f"pg {page + 1}/{total_pages}"
        _rx(draw, H - FOOTER_H + 2, page_txt, F["small"], MUTED)

    return img


# ── display thread ────────────────────────────────────────────────────────────
def display_loop(state_dict):
    """
    Start as a daemon thread from detect.py:
        threading.Thread(target=screen.display_loop, args=(state,), daemon=True).start()
    """
    global _fb

    _load_fonts()

    _fb = _find_fb()
    if _fb is None:
        print("[screen] No framebuffer found — display disabled.")
        return
    print(f"[screen] Using {_fb}  ({W}×{H})")

    # Backlight on
    try:
        import RPi.GPIO as GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(PIN_LED, GPIO.OUT)
        pwm = GPIO.PWM(PIN_LED, 1000)
        pwm.start(100)
        print(f"[screen] Backlight on (GPIO {PIN_LED})")
    except Exception as e:
        print(f"[screen] Backlight: {e}")

    page            = 0
    last_page_flip  = time.time()
    tick            = 0

    while True:
        try:
            # Thread-safe state copy
            with state_dict["lock"]:
                snap = {
                    "locked":        state_dict["locked"],
                    "current_user":  state_dict["current_user"],
                    "missing_tools": list(state_dict["missing_tools"]),
                    "slots": {k: dict(v) for k, v in state_dict["slots"].items()},
                }

            # Page flip
            n_slots = len(snap["slots"])
            if n_slots > PAGE_SIZE and time.time() - last_page_flip > PAGE_SECS:
                page           = (page + 1) % max(1, (n_slots + PAGE_SIZE - 1) // PAGE_SIZE)
                last_page_flip = time.time()

            pulse = (tick % 2 == 1)   # blink missing dot every other render
            img   = render(snap, page=page, pulse=pulse)
            _write(img)

        except Exception as e:
            print(f"[screen] error: {e}")

        tick += 1
        time.sleep(0.5)
