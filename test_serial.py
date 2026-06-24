import serial
import time

ser = serial.Serial("/dev/ttyUSB0", 115200)

time.sleep(2)

ser.write(b"VEL -500 500\n")

time.sleep(10)

