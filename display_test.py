#!/usr/bin/env python3
"""
display_test.py  —  Colour-flash test for JC2432S028 TFT (ILI9341)
Writes directly to the Linux framebuffer (/dev/fb0) using the kernel
fbtft driver that is already running (fb_ili9341, 320×240 landscape).

No luma.lcd needed — only PIL and numpy (already in requirements).

Wiring confirmed by dmesg:
  spi0.0 → ILI9341 display  → /dev/fb0  (320×240, RGB565, 32 MHz)
  spi0.1 → ADS7846 touchscreen
  GPIO18 → LED backlight (PWM)

If colours appear with red/blue swapped, set BGR = True below.
"""
import glob
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── config ────────────────────────────────────────────────────────────────────
# Kernel rotated 90° → landscape framebuffer
WIDTH, HEIGHT = 320, 240

PIN_LED = 18   # backlight PWM

BGR = False    # flip to True if red and blue appear swapped


# ── framebuffer ───────────────────────────────────────────────────────────────
def find_fb():
    """Return the framebuffer device path for the ILI9341."""
    for sysfs in sorted(glob.glob("/sys/class/graphics/fb*")):
        try:
            name = open(f"{sysfs}/name").read().strip()
            if "ili9341" in name.lower():
                return f"/dev/{os.path.basename(sysfs)}"
        except OSError:
            pass
    # Fallback: first existing fb device
    for dev in ("/dev/fb0", "/dev/fb1"):
        if os.path.exists(dev):
            return dev
    return "/dev/fb0"


FB = find_fb()


def show(img: Image.Image):
    """Write a PIL RGB image to the framebuffer as packed RGB565."""
    if img.size != (WIDTH, HEIGHT):
        img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)
    arr = np.array(img.convert("RGB"), dtype=np.uint16)
    if BGR:
        r, g, b = arr[:, :, 2], arr[:, :, 1], arr[:, :, 0]
    else:
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    try:
        with open(FB, "wb") as f:
            f.write(rgb565.astype(np.uint16).tobytes())
    except PermissionError:
        print(f"  Permission denied on {FB}.")
        print("  Fix with:  sudo usermod -a -G video $USER  (then re-login)")
        print("  Or run:    sudo python display_test.py")
        sys.exit(1)


# ── backlight ─────────────────────────────────────────────────────────────────
def backlight_on():
    try:
        import RPi.GPIO as GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(PIN_LED, GPIO.OUT)
        pwm = GPIO.PWM(PIN_LED, 1000)
        pwm.start(100)
        return pwm
    except Exception as e:
        print(f"  [backlight] {e} — continuing")
        return None


def backlight_off(pwm):
    try:
        if pwm:
            pwm.stop()
        import RPi.GPIO as GPIO
        GPIO.cleanup()
    except Exception:
        pass


# ── drawing helpers ───────────────────────────────────────────────────────────
def _font(size, bold=False):
    suffix = "-Bold" if bold else ""
    for path in (
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf",
        f"/usr/share/fonts/truetype/liberation/LiberationSans{'-Bold' if bold else '-Regular'}.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def text_w(draw, text, font):
    try:
        bb = draw.textbbox((0, 0), text, font=font)
        return bb[2] - bb[0]
    except AttributeError:
        return draw.textsize(text, font=font)[0]


def center(draw, y, text, font, fill):
    x = (WIDTH - text_w(draw, text, font)) // 2
    draw.text((x, y), text, fill=fill, font=font)


# ── test: colour flash ────────────────────────────────────────────────────────
COLORS = [
    ("Red",     (220,  50,  50)),
    ("Green",   ( 50, 200,  50)),
    ("Blue",    ( 50,  50, 220)),
    ("Yellow",  (230, 220,  40)),
    ("Cyan",    ( 40, 210, 210)),
    ("Magenta", (210,  40, 210)),
    ("White",   (240, 240, 240)),
    ("Black",   (  0,   0,   0)),
    ("Orange",  (240, 130,  20)),
]

def test_flash(delay=0.5):
    font = _font(26, bold=True)
    for name, rgb in COLORS:
        print(f"    {name}")
        img  = Image.new("RGB", (WIDTH, HEIGHT), rgb)
        draw = ImageDraw.Draw(img)
        luma = rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114
        ink  = (0, 0, 0) if luma > 140_000 else (255, 255, 255)
        center(draw, HEIGHT // 2 - 15, name, font, ink)
        show(img)
        time.sleep(delay)


# ── test: gradient ────────────────────────────────────────────────────────────
def test_gradient():
    img  = Image.new("RGB", (WIDTH, HEIGHT))
    data = []
    for y in range(HEIGHT):
        t = y / HEIGHT
        data += [(int(220 * (1 - t)), 0, int(220 * t))] * WIDTH
    img.putdata(data)
    draw = ImageDraw.Draw(img)
    center(draw, HEIGHT // 2 - 12, "Gradient OK", _font(18, bold=True), (255, 255, 255))
    show(img)
    time.sleep(1.0)


# ── tool-chest UI ─────────────────────────────────────────────────────────────
BG      = ( 15,  17,  21)
BLUE    = ( 59, 130, 246)
GREEN   = ( 74, 222, 128)
RED     = (248, 113, 113)
YELLOW  = (250, 204,  21)
MUTED   = (100, 116, 139)
DIVIDER = ( 45,  55,  72)
WHITE   = (226, 232, 240)


def draw_toolchest_screen(slots, locked, user, missing_names):
    """
    Build the tool-chest status image (320×240 landscape).

    slots         — list of {"name": str, "present": bool}
    locked        — bool
    user          — str  active user or "—"
    missing_names — list of slot names that are OUT
    """
    img  = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    f15b = _font(15, bold=True)
    f12b = _font(12, bold=True)
    f12  = _font(12)
    f10  = _font(10)

    # ── header ────────────────────────────────────────────────────────────
    draw.rectangle([(0, 0), (WIDTH, 28)], fill=BLUE)
    center(draw, 7, "Smart Tool Chest", f15b, WHITE)

    # ── split layout: left panel | slot grid ─────────────────────────────
    SPLIT = 130   # x where the grid starts

    # Lock badge
    badge_col = RED if locked else GREEN
    badge_txt = "LOCKED" if locked else "UNLOCKED"
    draw.rectangle([(4, 34), (SPLIT - 6, 54)], fill=DIVIDER, outline=badge_col)
    tw = text_w(draw, badge_txt, f12b)
    draw.text(((SPLIT - 6 + 4 - tw) // 2, 39), badge_txt, fill=badge_col, font=f12b)

    # Info rows
    rows = [
        ("User",    user),
        ("Missing", str(len(missing_names))),
    ]
    y = 60
    for label, val in rows:
        draw.text((6,  y), label + ":", fill=MUTED, font=f10)
        draw.text((60, y), val,         fill=WHITE, font=f10)
        y += 16

    # Missing list
    if missing_names:
        draw.text((6, y + 4), "OUT:", fill=RED, font=f10)
        for n in missing_names[:4]:
            y += 14
            draw.text((10, y), "• " + n, fill=RED, font=f10)
    else:
        draw.text((6, y + 4), "All present", fill=GREEN, font=f10)

    # ── slot grid (right panel) ───────────────────────────────────────────
    COLS   = 3
    cell_w = (WIDTH - SPLIT - 8) // COLS
    cell_h = 26
    grid_y = 34

    draw.line([(SPLIT - 2, 30), (SPLIT - 2, HEIGHT)], fill=DIVIDER, width=1)

    for i, slot in enumerate(slots[: COLS * ((HEIGHT - grid_y) // cell_h)]):
        col  = i % COLS
        row  = i // COLS
        x0   = SPLIT + col * cell_w + 2
        y0   = grid_y + row * cell_h + 2
        x1   = x0 + cell_w - 3
        y1   = y0 + cell_h - 3
        c    = GREEN if slot["present"] else RED
        draw.rectangle([(x0, y0), (x1, y1)], fill=DIVIDER, outline=c)
        # dot
        draw.ellipse([(x0 + 3, y0 + 7), (x0 + 11, y0 + 15)], fill=c)
        draw.text((x0 + 14, y0 + 7), slot["name"][:5], fill=WHITE, font=f10)

    # ── footer ────────────────────────────────────────────────────────────
    draw.line([(0, HEIGHT - 14), (WIDTH, HEIGHT - 14)], fill=DIVIDER, width=1)
    draw.text((4,   HEIGHT - 11), time.strftime("%H:%M"), fill=MUTED, font=f10)
    draw.text((WIDTH - 80, HEIGHT - 11), "tool-chest v1", fill=DIVIDER, font=f10)

    return img


def test_ui_mockup():
    demo_slots = [
        {"name": "loctite",   "present": True},
        {"name": "pliers",    "present": False},
        {"name": "strippers", "present": True},
        {"name": "drill",     "present": True},
        {"name": "wrench",    "present": False},
        {"name": "tape",      "present": True},
    ]
    missing = [s["name"] for s in demo_slots if not s["present"]]
    img = draw_toolchest_screen(demo_slots, locked=True, user="bert",
                                missing_names=missing)
    show(img)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 46)
    print("  Smart Tool Chest — TFT Display Test")
    print(f"  Framebuffer: {FB}   ({WIDTH}×{HEIGHT} landscape)")
    print("=" * 46)

    print(f"\n[1] Backlight on (GPIO {PIN_LED}) …")
    pwm = backlight_on()

    print(f"[2] Writing to {FB} …")
    # Quick black fill to confirm we can open the device
    show(Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0)))
    print("  OK\n")

    print("[3] Colour flash (0.5 s each) …")
    test_flash(delay=0.5)

    print("[4] Gradient test …")
    test_gradient()

    print("[5] Tool-chest UI mockup …")
    test_ui_mockup()

    print("\n  All tests passed — UI mockup on screen.")
    print("  Ctrl-C to exit.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    print("Cleaning up …")
    backlight_off(pwm)


if __name__ == "__main__":
    main()
