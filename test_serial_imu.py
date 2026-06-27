import serial
import time

ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.2)
time.sleep(2)

def send(cmd):
    print("SEND:", cmd)
    ser.write((cmd + "\n").encode())
    time.sleep(0.2)
    while ser.in_waiting:
        print(ser.readline().decode(errors="ignore").strip())

send("STOP")
send("SET_BASE 3500")
send("SET_IMU_MAX 250")
send("SET_IMU_KP 45")
send("IMU RECAL")

# keep robot perfectly still during this
time.sleep(4)

send("LOCK_HEADING_GO")
time.sleep(1)

# forward test
send("VEL 3500 3500")

start = time.time()
while time.time() - start < 8:
    ser.write(b"STATUS\n")
    time.sleep(0.25)
    while ser.in_waiting:
        print(ser.readline().decode(errors="ignore").strip())

send("STOP")
