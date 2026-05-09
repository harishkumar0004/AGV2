#!/usr/bin/env python3
"""
Raspberry Pi master for the Arduino Mega AGV forward/stop test.

Usage:
  python3 raspberry_pi_forward_stop.py --port /dev/ttyACM0

Type:
  f or forward  -> send FORWARD
  s or stop     -> send STOP
  q or quit     -> send STOP and exit
"""

import argparse
import sys
import threading
import time

import serial


def serial_reader(ser: serial.Serial, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            line = ser.readline().decode("utf-8", errors="replace").strip()
        except serial.SerialException as exc:
            print(f"Serial read error: {exc}", file=sys.stderr)
            stop_event.set()
            return

        if line:
            print(f"[MEGA] {line}")


def send_command(ser: serial.Serial, command: str) -> None:
    ser.write((command.strip().upper() + "\n").encode("ascii"))
    ser.flush()
    print(f"[PI] {command.strip().upper()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="AGV forward/stop serial master")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Arduino Mega serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    args = parser.parse_args()

    stop_event = threading.Event()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.2)
    except serial.SerialException as exc:
        print(f"Could not open {args.port}: {exc}", file=sys.stderr)
        return 1

    with ser:
        time.sleep(2.0)  # Arduino resets when USB serial opens.
        ser.reset_input_buffer()

        reader = threading.Thread(target=serial_reader, args=(ser, stop_event), daemon=True)
        reader.start()

        print("Waiting for Mega. Keep AGV still during gyro calibration.")
        print("Commands: f=FORWARD, s=STOP, q=STOP+QUIT")

        try:
            while not stop_event.is_set():
                user_input = input("> ").strip().lower()

                if user_input in ("f", "forward"):
                    send_command(ser, "FORWARD")
                elif user_input in ("s", "stop"):
                    send_command(ser, "STOP")
                elif user_input in ("q", "quit", "exit"):
                    send_command(ser, "STOP")
                    break
                elif user_input:
                    print("Use f, s, or q.")
        except KeyboardInterrupt:
            print()
            send_command(ser, "STOP")
        finally:
            stop_event.set()
            reader.join(timeout=1.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

