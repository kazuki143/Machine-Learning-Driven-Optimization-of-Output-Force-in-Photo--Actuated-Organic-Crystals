import serial
import time

# Moving the stage for the experiment
def control_serial1():
    port = 'COM3'
    baudrate = 9600
    with serial.Serial(port, baudrate, timeout=1) as ser:
        ser.write(b'M:W-P30000-P10000\r\n')
        time.sleep(0.1)
        ser.write(b'G:\r\n')
        time.sleep(7)
        ser.write(b'Q:\r\n')
        response = ser.read_all()
        print("Positive move response:", response.decode())

# Moving the stage back to its original position after the experiment
def control_serial2():
    port = 'COM3'
    baudrate = 9600
    with serial.Serial(port, baudrate, timeout=1) as ser:
        ser.write(b'M:W+P50000+P50000\r\n')
        time.sleep(15)
        ser.write(b'G:\r\n')
        time.sleep(7)
        ser.write(b'Q:\r\n')
        response = ser.read_all()
        print("Negative move response:", response.decode())

if __name__ == '__main__':
    control_serial1()
    control_serial2()
