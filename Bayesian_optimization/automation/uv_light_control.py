import serial
import time

# Control of irradiation intensity
def uv_light_serial(intensity=1,num=1):
    """
    Control the irradiation intensity of a UV light source via serial communication.

    Input:
    --------
    intensity (int, optional (default=1)): The irradiation intensity is specified as a percentage (0-100%). A value of 100 corresponds to the maximum possible irradiation intensity.
    num (int, optional (default=1)): The channel number for the UV light source.

    Returns:
    --------
    None
    """

    serial_port = "COM5"  # Specify the communication port
    baud_rate = 9600      # Set baud rate
    timeout = 5           # Timeout in seconds
    irradiation_time = 300

    ser = serial.Serial(serial_port, baud_rate, timeout=timeout)
    command = f">LTSET-{num}-{irradiation_time}\r"
    ser.write(command.encode())
    command = f">PARASET-{num}-{intensity}\r"
    ser.write(command.encode())
    command = f">ON-{num}\r"
    ser.write(command.encode())

    time.sleep(1)
    response = ser.readline()
    print(response.decode())
    ser.close()

if __name__ == '__main__':
    intensity=1
    uv_light_serial(intensity=intensity,num=2)
