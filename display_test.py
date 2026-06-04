#!/usr/bin/env python3
"""
display_test.py  —  Colour-flash test for JC2432S028 TFT (ILI9341, 240×320)
Confirms wiring and SPI, then shows a tool-chest UI mockup as the template
for the eventual live screen.

Wiring (BCM pin numbering):
  CS    → GPIO  8  (SPI0 CE0,  Pi pin 24)
  RESET → GPIO 25  (Pi pin 22)
  DC    → GPIO 24  (Pi pin 18)
  MOSI  → GPIO 10  (SPI0 MOSI, Pi pin 19)
  SCK   → GPIO 11  (SPI0 CLK,  Pi pin 23)
  MISO  → GPIO  9  (SPI0 MISO, Pi pin 21)  — passive, not driven by this code
  LED   → GPIO 18  (Pi pin 12) — PWM backlight

Setup:
  sudo raspi-config → Interface Options → SPI → Enable
  pip install luma.lcd Pillow RPi.GPIO spidev
"""
import sys
import time
from PIL import Image, ImageDraw, ImageFont

# ── hardware constants ────────────────────────────────────────────────────────
WIDTH,  HEIGHT  = 240, 320
PIN_DC, PIN_RST = 24, 25
PIN_LED         = 18
SPI_PORT        = 0
SPI_DEV         = 0   # CE0 = GPIO 8

# ILI9341 SPI clock — 32 MHz is reliable on Pi 3/4/5;
# drop to 16_000_000 if you see corrupted pixels.
SPI_HZ = 32_000_000


# ── backlight ─────────────────────────────────────────────────────────────────
def backlight_on():
    try:
        import RPi.GPIO as GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(PIN_LED, GPIO.OUT)
        pwm = GPIO.PWM(PIN_LED, 1000)   # 1 kHz
        pwm.start(100)                  # 100 % brightness
        return pwm
    except Exception as e:
        print(f"[backlight] {e} — continuing without PWM control")
        return None


def backlight_off(pwm):
    try:
        if pwm:
            pwm.stop()
        import RPi.GPIO as GPIO
        GPIO.cleanup()
    except Exception:
        pass


# ── display init ──────────────────────────────────────────────────────────────
def init_display():
    from luma.core.interface.serial import spi
    from luma.lcd.device import ili9341
    serial = spi(
        port=SPI_PORT,
        device=SPI_DEV,
        gpio_DC=PIN_DC,
        gpio_RST=PIN_RST,
        bus_speed_hz=SPI_HZ,
    )
    # bgr=False for RGB; if colours look wrong (e.g. red ↔ blue swapped) try bgr=True
    return ili9341(serial, width=WIDTH, height=HEIGHT, rotate=0, bgr=False)


# ── drawing helpers ───────────────────────────────────────────────────────────
def _font(size, bold=False):
    suffix = "-Bold" if bold else ""
    candidates = [
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf",
        f"/usr/share/fonts/truetype/liberation/LiberationSans{'-Bold' if bold else '-Regular'}.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_w(draw, text, font):
    try:
        bb = draw.textbbox((0, 0), text, font=font)
        return bb[2] - bb[0]
    except AttributeError:           # Pillow < 9.2
        return draw.textsize(text, font=font)[0]


def center_text(draw, y, text, font, fill):
    x = (WIDTH - text_w(draw, text, font)) // 2
    draw.text((x, y), text, fill=fill, font=font)


# ── colour flash ──────────────────────────────────────────────────────────────
FLASH_COLORS = [
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

def test_flash(device, delay=0.5):
    font = _font(26, bold=True)
    for name, rgb in FLASH_COLORS:
        print(f"    {name}")
        img  = Image.new("RGB", (WIDTH, HEIGHT), rgb)
        draw = ImageDraw.Draw(img)
        luma  = rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114
        ink   = (0, 0, 0) if luma > 140_000 else (255, 255, 255)
        center_text(draw, HEIGHT // 2 - 15, name, font, ink)
        device.display(img)
        time.sleep(delay)


# ── gradient ──────────────────────────────────────────────────────────────────
def test_gradient(device):
    img  = Image.new("RGB", (WIDTH, HEIGHT))
    data = []
    for y in range(HEIGHT):
        t = y / HEIGHT
        data += [(int(220 * (1 - t)), 0, int(220 * t))] * WIDTH
    img.putdata(data)
    draw = ImageDraw.Draw(img)
    center_text(draw, HEIGHT // 2 - 14, "Gradient", _font(20, bold=True), (255, 255, 255))
    device.display(img)
    time.sleep(1.0)


# ── tool-chest UI mockup ──────────────────────────────────────────────────────
# Colours that match the web dashboard (dark theme)
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
    Build a PIL image representing the tool-chest status.
    Call device.display(img) with the result.

    slots         — list of {"name": str, "present": bool}
    locked        — bool
    user          — str  (current user or "—")
    missing_names — list of slot names that are OUT
    """
    img  = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    f16b = _font(16, bold=True)
    f13b = _font(13, bold=True)
    f13  = _font(13)
    f11  = _font(11)

    # ── header bar ────────────────────────────────────────────────────────
    draw.rectangle([(0, 0), (WIDTH, 34)], fill=BLUE)
    center_text(draw, 9, "Smart Tool Chest", f16b, WHITE)

    # ── lock badge ────────────────────────────────────────────────────────
    badge_col = RED if locked else GREEN
    badge_txt = "LOCKED" if locked else "UNLOCKED"
    draw.rectangle([(6, 40), (WIDTH - 6, 62)], fill=DIVIDER, outline=badge_col)
    center_text(draw, 47, badge_txt, f13b, badge_col)

    # ── user row ──────────────────────────────────────────────────────────
    y = 70
    draw.text((10, y), "User:", fill=MUTED, font=f11)
    draw.text((54, y), user, fill=WHITE, font=f11)
    y += 16
    draw.line([(0, y), (WIDTH, y)], fill=DIVIDER, width=1)

    # ── missing banner ────────────────────────────────────────────────────
    y += 4
    miss_n = len(missing_names)
    if miss_n:
        draw.rectangle([(6, y), (WIDTH - 6, y + 18)], fill=(80, 20, 20), outline=RED)
        txt = f"{miss_n} MISSING: " + ", ".join(missing_names[:3])
        if miss_n > 3:
            txt += "…"
        draw.text((10, y + 3), txt, fill=RED, font=f11)
    else:
        draw.rectangle([(6, y), (WIDTH - 6, y + 18)], fill=(10, 50, 20), outline=GREEN)
        draw.text((10, y + 3), "All tools present", fill=GREEN, font=f11)
    y += 24

    # ── slot grid ─────────────────────────────────────────────────────────
    draw.line([(0, y), (WIDTH, y)], fill=DIVIDER, width=1)
    y += 6
    draw.text((10, y), "Tool Slots", fill=MUTED, font=f11)
    y += 14

    COLS     = 3
    cell_w   = (WIDTH - 12) // COLS
    cell_h   = 30
    max_rows = (HEIGHT - y - 20) // cell_h

    for idx, slot in enumerate(slots[:COLS * max_rows]):
        row  = idx // COLS
        col  = idx  % COLS
        x0   = 6  + col * cell_w
        y0   = y  + row * cell_h
        x1   = x0 + cell_w - 3
        y1   = y0 + cell_h - 3
        col_ = GREEN if slot["present"] else RED
        draw.rectangle([(x0, y0), (x1, y1)], fill=DIVIDER, outline=col_)
        # Status dot
        dot_x = x0 + 6; dot_y = y0 + 7
        draw.ellipse([(dot_x, dot_y), (dot_x + 8, dot_y + 8)], fill=col_)
        # Slot name (truncate if too long)
        name = slot["name"][:7]
        draw.text((dot_x + 12, y0 + 8), name, fill=WHITE, font=f11)

    # ── footer ────────────────────────────────────────────────────────────
    draw.line([(0, HEIGHT - 16), (WIDTH, HEIGHT - 16)], fill=DIVIDER, width=1)
    ts = time.strftime("%H:%M")
    draw.text((10, HEIGHT - 13), ts, fill=MUTED, font=f11)

    return img


def test_ui_mockup(device):
    # Demo data — replace with live state from detect.py later
    demo_slots = [
        {"name": "loctite",  "present": True},
        {"name": "pliers",   "present": False},
        {"name": "strippers","present": True},
        {"name": "drill",    "present": True},
        {"name": "wrench",   "present": False},
        {"name": "tape",     "present": True},
    ]
    missing = [s["name"] for s in demo_slots if not s["present"]]
    img = draw_toolchest_screen(demo_slots, locked=True, user="bert", missing_names=missing)
    device.display(img)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 46)
    print("  Smart Tool Chest — TFT Display Test")
    print("  JC2432S028  /  ILI9341  /  240×320  SPI")
    print("=" * 46)

    print("\n[1] Backlight on (GPIO 18) …")
    pwm = backlight_on()

    print("[2] Initialising ILI9341 over SPI …")
    try:
        device = init_display()
    except Exception as e:
        print(f"\n  FAILED: {e}")
        print("\n  Checklist:")
        print("  • SPI enabled?   sudo raspi-config → Interface Options → SPI")
        print("  • Libraries?     pip install luma.lcd Pillow RPi.GPIO spidev")
        print("  • Wiring ok?     See pinout at top of this file")
        backlight_off(pwm)
        sys.exit(1)
    print("  OK\n")

    print("[3] Colour flash (0.5 s each) …")
    test_flash(device, delay=0.5)

    print("[4] Gradient sweep …")
    test_gradient(device)

    print("[5] Tool-chest UI mockup …")
    test_ui_mockup(device)

    print("\n  All tests passed — UI mockup is on screen.")
    print("  Ctrl-C to exit.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    print("\nCleaning up …")
    backlight_off(pwm)
    print("Done.")


if __name__ == "__main__":
    main()
