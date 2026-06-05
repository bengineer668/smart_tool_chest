#!/usr/bin/env python3
"""
rfid_register.py — Register RFID cards in toolchest_config.json.
Scan a card, enter a name, and it's saved as an authorized user.

Wiring: SPI1/CE2 (GPIO16), RST=GPIO17
Enable SPI1 first: add 'dtoverlay=spi1-3cs' to /boot/firmware/config.txt
"""
import json
import os
import sys
import time

RFID_FILE = "rfid_cards.json"


def load_rfid():
    if os.path.exists(RFID_FILE) and os.path.getsize(RFID_FILE) > 2:
        with open(RFID_FILE) as f:
            return json.load(f)
    # One-time migration: pull cards out of toolchest_config.json if present
    if os.path.exists("toolchest_config.json"):
        try:
            with open("toolchest_config.json") as f:
                cards = json.load(f).get("rfid_cards", {})
            if cards:
                print(f"[migrate] Moving {len(cards)} card(s) to {RFID_FILE}")
                save_rfid(cards)
                return cards
        except Exception:
            pass
    return {}


def save_rfid(cards):
    with open(RFID_FILE, "w") as f:
        json.dump(cards, f, indent=2)


def wait_for_card(reader):
    while True:
        (status, _) = reader.MFRC522_Request(reader.PICC_REQIDL)
        if status == reader.MI_OK:
            (status, uid) = reader.MFRC522_Anticoll()
            if status == reader.MI_OK:
                return '-'.join(str(b) for b in uid)
        time.sleep(0.1)


def main():
    try:
        from mfrc522 import MFRC522
        reader = MFRC522(bus=1, device=0, pin_rst=-1, pin_mode=11)
    except ImportError:
        print("[ERROR] mfrc522 not installed. Run: pip install mfrc522")
        sys.exit(1)
    except FileNotFoundError:
        print("[ERROR] /dev/spidev1.2 not found — SPI1 not enabled.")
        print("  Fix: add 'dtoverlay=spi1-3cs' to /boot/firmware/config.txt and reboot.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Could not init RC522: {e}")
        sys.exit(1)

    rfid_cards = load_rfid()

    print("RC522 RFID Card Registration")
    print("=" * 40)
    if rfid_cards:
        print("Registered cards:")
        for uid, name in rfid_cards.items():
            print(f"  {uid}  →  {name}")
    else:
        print("No cards registered yet.")
    print()

    while True:
        print("Place a card on the reader (Ctrl+C to quit)...")
        uid_str = wait_for_card(reader)

        print(f"\nCard UID: {uid_str}")
        if uid_str in rfid_cards:
            print(f"  Currently: {rfid_cards[uid_str]}")

        name = input("Name for this card (Enter to skip): ").strip()
        if name:
            rfid_cards[uid_str] = name
            save_rfid(rfid_cards)
            print(f"  Saved: {uid_str}  →  {name}")
        else:
            print("  Skipped.")

        again = input("\nRegister another? (y/n): ").strip().lower()
        if again != "y":
            break

    print("\nFinal registered cards:")
    for uid, name in rfid_cards.items():
        print(f"  {uid}  →  {name}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDone.")
