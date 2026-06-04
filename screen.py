#!/usr/bin/env python3
"""
screen.py — TFT display for FRC Team 668 · Apes of Wrath · Smart Tool Chest
Renders live tool status to the ILI9341 framebuffer (/dev/fb0).
320×240 landscape (kernel fbtft applies 90° rotation).

Drop a file named  team668.png  in the project folder to use it as the
boot logo.  If no file is found a procedurally drawn gorilla is used instead.
"""
import glob
import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── display constants ─────────────────────────────────────────────────────────
W, H      = 320, 240
PIN_LED   = 18
BGR       = False       # flip if red and blue appear swapped
PAGE_SIZE = 5           # tool rows per screen page
PAGE_SECS = 4.0         # seconds before auto-advancing to next page
LOGO_PATH = "team668.png"

# ── Team 668 colour palette ───────────────────────────────────────────────────
BG          = (  8,   6,   4)   # near-black, warm tint
HEADER_BG   = ( 55,  20,   0)   # dark burnt-orange
ORANGE      = (255, 107,   0)   # team primary
ORANGE_MID  = (180,  70,   0)   # mid orange for details
GREEN       = ( 80, 220,  90)   # tool present
RED         = (240,  60,  55)   # tool missing / locked
MUTED       = (130, 100,  70)   # secondary text (warm grey)
WHITE       = (235, 220, 205)   # warm white
DIVIDER     = ( 35,  22,  10)   # warm dark divider line
PILL_LOCK   = (( 90,  12,   8), (252, 165, 165))   # (bg, text)
PILL_UNLOCK = (( 90,  45,   0), (255, 185,  80))   # orange unlocked
ALERT_MISS  = (( 55,   8,   5), (240,  80,  70))   # (bg, text)
ALERT_OK    = (( 10,  42,  12), ( 80, 220,  90))
ROW_MISS_BG = ( 42,   7,   5)

# ── framebuffer ───────────────────────────────────────────────────────────────
_fb = None


def _find_fb():
    for sysfs in sorted(glob.glob("/sys/class/graphics/fb*")):
        try:
            if "ili9341" in open(f"{sysfs}/name").read().lower():
                return f"/dev/{os.path.basename(sysfs)}"
        except OSError:
            pass
    for dev in ("/dev/fb0", "/dev/fb1"):
        if os.path.exists(dev):
            return dev
    return None


def _write(img: Image.Image):
    if not _fb:
        return
    a = np.array(img.convert("RGB"), dtype=np.uint16)
    r, g, b = (a[:, :, 2], a[:, :, 1], a[:, :, 0]) if BGR \
         else (a[:, :, 0], a[:, :, 1], a[:, :, 2])
    try:
        with open(_fb, "wb") as f:
            f.write((((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3))
                    .astype(np.uint16).tobytes())
    except Exception:
        pass


# ── fonts ─────────────────────────────────────────────────────────────────────
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

    F["num"]    = font(30, bold=True)   # "668" on splash
    F["big"]    = font(17, bold=True)   # "APES OF WRATH" on splash
    F["head"]   = font(14, bold=True)   # header title
    F["chip"]   = font(9,  bold=True)   # small badge text
    F["label"]  = font(13, bold=True)   # user / alert row
    F["tool"]   = font(13)              # tool names
    F["small"]  = font(10)              # footer, overflow note


# ── drawing helpers ───────────────────────────────────────────────────────────
def _tw(draw, text, font):
    try:
        bb = draw.textbbox((0, 0), text, font=font)
        return bb[2] - bb[0]
    except AttributeError:
        return draw.textsize(text, font=font)[0]


def _cx(draw, y, text, font, fill, x0=0, width=W):
    """Centre text horizontally within [x0, x0+width]."""
    draw.text((x0 + (width - _tw(draw, text, font)) // 2, y),
              text, fill=fill, font=font)


def _rx(draw, y, text, font, fill, margin=8):
    draw.text((W - _tw(draw, text, font) - margin, y), text, fill=fill, font=font)


def _pill(draw, x1, y1, x2, y2, bg, text, font, fg):
    try:
        draw.rounded_rectangle([(x1, y1), (x2, y2)], radius=4, fill=bg)
    except AttributeError:
        draw.rectangle([(x1, y1), (x2, y2)], fill=bg)
    tw = _tw(draw, text, font)
    draw.text((x1 + ((x2 - x1) - tw) // 2, y1 + ((y2 - y1) - 10) // 2),
              text, fill=fg, font=font)


def _hline(draw, y, color=None):
    draw.line([(0, y), (W, y)], fill=color or DIVIDER, width=1)


# ── gorilla head ──────────────────────────────────────────────────────────────
def _ape(draw, cx, cy, r):
    """
    Procedurally draw a gorilla head centred at (cx, cy) with radius r.
    Scales cleanly from r=14 (header icon) to r=62 (splash screen).
    """
    cx, cy, r = int(cx), int(cy), int(r)

    # Head silhouette
    draw.ellipse([cx-r, cy-int(r*1.05), cx+r, cy+int(r*0.9)],
                 fill=(30, 17, 6))
    # Sagittal crest / brow ridge — orange arc across the top
    br = int(r * 0.85)
    draw.arc([cx-br, cy-int(r*1.05), cx+br, cy-int(r*0.12)],
             start=180, end=360, fill=ORANGE, width=max(2, r // 6))
    # Lighter muzzle region
    fr = int(r * 0.66)
    draw.ellipse([cx-fr, cy-int(r*0.22), cx+fr, cy+int(r*0.82)],
                 fill=(55, 32, 12))
    # Eyes
    er = max(2, r // 5)
    ey = cy - int(r * 0.18)
    for ex in (cx - int(r * 0.36), cx + int(r * 0.36)):
        draw.ellipse([ex-er, ey-er, ex+er, ey+er], fill=(200, 200, 200))
        draw.ellipse([ex-er//2, ey-er//2, ex+er//2, ey+er//2], fill=(6, 6, 6))
    # Nose
    nr = max(2, r // 6)
    ny = cy + int(r * 0.2)
    draw.ellipse([cx-nr, ny, cx+nr, ny+nr+2], fill=(24, 13, 4))
    nl = max(1, nr // 2)
    draw.ellipse([cx-nr+1, ny+2, cx-1,     ny+nl+3], fill=(8, 4, 0))
    draw.ellipse([cx+1,    ny+2, cx+nr-1,  ny+nl+3], fill=(8, 4, 0))
    # Smile
    if r >= 9:
        mw = int(r * 0.5)
        my = cy + int(r * 0.5)
        draw.arc([cx-mw, my-mw//4, cx+mw, my+mw//4],
                 start=0, end=180, fill=(24, 13, 4), width=max(1, r // 10))
    # Orange outline
    draw.arc([cx-r, cy-int(r*1.05), cx+r, cy+int(r*0.9)],
             start=0, end=360, fill=ORANGE, width=max(1, r // 15))


def _load_logo(size):
    """Try to load team668.png; return RGBA Image or None."""
    try:
        return Image.open(LOGO_PATH).convert("RGBA").resize(size, Image.LANCZOS)
    except Exception:
        return None


# ── splash screen (boot) ──────────────────────────────────────────────────────
def _splash() -> Image.Image:
    img  = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Orange border frame
    draw.rectangle([(1, 1), (W-2, H-2)], outline=ORANGE, width=2)
    draw.rectangle([(4, 4), (W-5, H-5)], outline=ORANGE_MID, width=1)

    # ── Left panel: logo / drawn ape ──────────────────────────────────────
    logo = _load_logo((130, 130))
    if logo:
        base = img.convert("RGBA")
        base.paste(logo, (10, (H - 130) // 2), logo)
        img  = base.convert("RGB")
        draw = ImageDraw.Draw(img)
    else:
        _ape(draw, 78, H // 2, 62)

    # Vertical divider
    draw.line([(W // 2, 10), (W // 2, H - 10)], fill=ORANGE_MID, width=1)

    # ── Right panel: team branding ────────────────────────────────────────
    rp_x  = W // 2 + 8       # right panel left edge  (168)
    rp_w  = W - 8 - rp_x     # right panel width      (144)

    def rp_cx(y, text, font, fill):
        draw.text((rp_x + (rp_w - _tw(draw, text, font)) // 2, y),
                  text, fill=fill, font=font)

    # "668"
    rp_cx(38,  "668",           F["num"],  ORANGE)
    # thin separator
    draw.line([(rp_x+4, 76), (W-12, 76)], fill=ORANGE_MID, width=1)
    # "APES OF WRATH"
    rp_cx(82,  "APES OF WRATH", F["big"],  WHITE)
    # thin separator
    draw.line([(rp_x+4, 105), (W-12, 105)], fill=DIVIDER, width=1)
    # "TEAM 668"
    rp_cx(110, "TEAM  668",     F["big"],  ORANGE)
    # thin separator
    draw.line([(rp_x+4, 133), (W-12, 133)], fill=DIVIDER, width=1)
    # subtitle
    rp_cx(139, "Smart Tool Chest", F["small"], MUTED)

    # "Initialising…" at bottom
    draw.text((rp_x + 4, H - 22), "Starting up...", fill=MUTED, font=F["small"])

    return img


# ── main status screen ────────────────────────────────────────────────────────
def render(snap, page=0, pulse=False) -> Image.Image:
    """Build one 320×240 status frame from a detect.py state snapshot."""
    locked  = snap.get("locked", True)
    user    = snap.get("current_user", "")
    missing = snap.get("missing_tools", [])
    slots   = snap.get("slots", {})

    user_disp = "—" if (not user or user.startswith("None")) else user

    img  = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # ── 1. HEADER (y 0–35) ───────────────────────────────────────────────
    draw.rectangle([(0, 0), (W, 35)], fill=HEADER_BG)

    # Ape icon on left (r=14 → 28×29 px)
    _ape(draw, 20, 18, 14)

    # Lock badge on right
    pill_bg, pill_fg = PILL_LOCK if locked else PILL_UNLOCK
    pill_txt = "LOCKED" if locked else "UNLOCKED"
    pill_w   = _tw(draw, pill_txt, F["chip"]) + 14
    _pill(draw, W - pill_w - 5, 8, W - 5, 26,
          pill_bg, pill_txt, F["chip"], pill_fg)

    # Title centred between icon and badge
    icon_right  = 38
    badge_left  = W - pill_w - 9
    title_space = badge_left - icon_right
    draw.text(
        (icon_right + (title_space - _tw(draw, "Smart Tool Chest", F["head"])) // 2, 11),
        "Smart Tool Chest", fill=WHITE, font=F["head"]
    )

    # "TEAM 668" micro-chip below title
    chip_txt = "TEAM 668"
    chip_x   = icon_right + (title_space - _tw(draw, chip_txt, F["chip"])) // 2
    draw.text((chip_x, 26), chip_txt, fill=ORANGE, font=F["chip"])

    _hline(draw, 36)

    # ── 2. USER ROW (y 37–57) ────────────────────────────────────────────
    draw.rectangle([(0, 37), (W, 57)], fill=DIVIDER)
    draw.text((10, 42), f"User:  {user_disp}", fill=WHITE, font=F["label"])
    _hline(draw, 58)

    # ── 3. ALERT BAR (y 59–80) ───────────────────────────────────────────
    n_miss   = len(missing)
    alert_bg, alert_fg = ALERT_MISS if n_miss else ALERT_OK
    draw.rectangle([(0, 59), (W, 80)], fill=alert_bg)

    if n_miss:
        names = ", ".join(missing)
        while _tw(draw, f"  ! Missing: {names}", F["label"]) > W - 12:
            names = names.rsplit(", ", 1)[0] + "..."
        alert_txt = f"  ! Missing: {names}"
    else:
        alert_txt = "  + All tools present"
    draw.text((8, 64), alert_txt, fill=alert_fg, font=F["label"])
    _hline(draw, 81)

    # ── 4. TOOL ROWS (y 82 downward, 24 px each) ─────────────────────────
    ROW_H    = 24
    FOOTER_H = 15
    MAX_Y    = H - FOOTER_H - 1

    slot_items  = list(slots.items())
    total_pages = max(1, (len(slot_items) + PAGE_SIZE - 1) // PAGE_SIZE)
    page        = page % total_pages
    visible     = slot_items[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]

    y = 82
    for name, info in visible:
        if y + ROW_H > MAX_Y:
            break
        present  = info.get("present", True)
        row_bg   = ROW_MISS_BG if not present else BG
        dot_col  = GREEN if present else (RED if not pulse else (100, 20, 18))
        name_col = WHITE if present else (255, 160, 155)
        stat_txt = "PRESENT" if present else "MISSING"
        stat_col = GREEN    if present else RED

        draw.rectangle([(0, y), (W, y + ROW_H - 1)], fill=row_bg)
        draw.ellipse([(9, y + 6), (20, y + 17)], fill=dot_col)
        draw.text((28, y + 5), name, fill=name_col, font=F["tool"])
        _rx(draw, y + 6, stat_txt, F["small"], stat_col)
        _hline(draw, y + ROW_H - 1)
        y += ROW_H

    # Overflow note
    overflow = len(slot_items) - (page + 1) * PAGE_SIZE
    if overflow > 0 and y + 14 <= MAX_Y:
        draw.text((28, y + 4),
                  f"+ {overflow} more  (page {page+1}/{total_pages})",
                  fill=MUTED, font=F["small"])

    # No-slots message
    if not slot_items:
        draw.text((28, 100), "No slots configured.", fill=MUTED, font=F["tool"])
        draw.text((28, 120), "Run calibrate.py first.", fill=MUTED, font=F["small"])

    # ── 5. FOOTER (bottom 15 px) ──────────────────────────────────────────
    _hline(draw, H - FOOTER_H)
    draw.rectangle([(0, H - FOOTER_H + 1), (W, H)], fill=DIVIDER)

    draw.text((6,  H - FOOTER_H + 3), time.strftime("%H:%M"), fill=MUTED, font=F["small"])
    n_ok = sum(1 for v in slots.values() if v.get("present", True))
    draw.text((50, H - FOOTER_H + 3), f"{n_ok}/{len(slots)} present", fill=MUTED, font=F["small"])

    if total_pages > 1:
        _rx(draw, H - FOOTER_H + 3, f"pg {page+1}/{total_pages}", F["small"], ORANGE_MID)
    else:
        _rx(draw, H - FOOTER_H + 3, "TEAM 668", F["small"], ORANGE_MID)

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
    print(f"[screen] Using {_fb}  ({W}×{H})  Team 668 theme")

    # Backlight
    try:
        import RPi.GPIO as GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(PIN_LED, GPIO.OUT)
        pwm = GPIO.PWM(PIN_LED, 1000)
        pwm.start(100)
    except Exception as e:
        print(f"[screen] Backlight: {e}")

    # Boot splash
    _write(_splash())
    time.sleep(2.5)

    page           = 0
    last_page_flip = time.time()
    tick           = 0

    while True:
        try:
            with state_dict["lock"]:
                snap = {
                    "locked":        state_dict["locked"],
                    "current_user":  state_dict["current_user"],
                    "missing_tools": list(state_dict["missing_tools"]),
                    "slots":         {k: dict(v) for k, v in state_dict["slots"].items()},
                }

            n = len(snap["slots"])
            if n > PAGE_SIZE and time.time() - last_page_flip > PAGE_SECS:
                page           = (page + 1) % max(1, (n + PAGE_SIZE - 1) // PAGE_SIZE)
                last_page_flip = time.time()

            _write(render(snap, page=page, pulse=(tick % 2 == 1)))

        except Exception as e:
            print(f"[screen] render error: {e}")

        tick += 1
        time.sleep(0.5)
