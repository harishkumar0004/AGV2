import serial
import time

ser = serial.Serial("/dev/ttyUSB0", 115200)

time.sleep(2)

ser.write(b"VEL -5000 -5000\n")

time.sleep(10)

