#!/usr/bin/env python3
"""
rfid_test.py — Beeps whenever an RFID card is detected.
Tests RC522 wiring and audio before running the main app.

Wiring: SPI1/CE2 (GPIO16), RST=GPIO17
Enable SPI1 first: add 'dtoverlay=spi1-3cs' to /boot/firmware/config.txt
"""
import os
import time

try:
    import pygame
    pygame.mixer.init()
    AUDIO_ENABLED = True
except Exception as e:
    AUDIO_ENABLED = False
    print(f"[audio] Not available: {e}")


def beep():
    if not AUDIO_ENABLED:
        print("  (audio not available)")
        return
    # Try rfid_beep first, fall back to missing_boop
    for name in ("rfid_beep", "missing_boop"):
        path = f"sounds/{name}.wav"
        if os.path.exists(path):
            sound = pygame.mixer.Sound(path)
            sound.set_volume(1.0)
            sound.play()
            return
    print("  (no sound file found in sounds/)")


def main():
    try:
        from mfrc522 import MFRC522
        reader = MFRC522(bus=1, device=2, pin_rst=17, pin_mode=11)
    except ImportError:
        print("[ERROR] mfrc522 not installed. Run: pip install mfrc522")
        return
    except FileNotFoundError:
        print("[ERROR] /dev/spidev1.2 not found — SPI1 not enabled.")
        print("  Fix: add 'dtoverlay=spi1-3cs' to /boot/firmware/config.txt and reboot.")
        return
    except Exception as e:
        print(f"[ERROR] Could not init RC522: {e}")
        return

    print("RFID Test  —  place cards on the reader (Ctrl+C to quit)")
    print("SPI1/CE2, RST=GPIO17")
    print()

    last_uid    = None
    last_uid_ts = 0.0

    while True:
        (status, _) = reader.MFRC522_Request(reader.PICC_REQIDL)
        if status == reader.MI_OK:
            (status, uid) = reader.MFRC522_Anticoll()
            if status == reader.MI_OK:
                uid_str = '-'.join(str(b) for b in uid)
                now = time.time()
                if uid_str != last_uid or (now - last_uid_ts) > 2.0:
                    last_uid    = uid_str
                    last_uid_ts = now
                    print(f"Card detected: {uid_str}")
                    beep()
        time.sleep(0.1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDone.")
