from glob import glob
import platform
from pydobot import Dobot

available_ports = glob('/dev/ttyUSB0')  # mask for Raspi Dobot port

device = Dobot(port=available_ports[0])
device.stop_conveyor_belt()
device.close()
