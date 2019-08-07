""" send PiCamera jpg stream.

A Raspberry Pi test program that uses imagezmq to send image frames from the
PiCamera continuously to a receiving program on a Mac that will display the
images as a video stream. Images are converted to jpg format before sending.

This program requires that the image receiving program be running first. Brief
test instructions are in that program: test_3_mac_receive_jpg.py.

Run with python client.py -s [socket address of server]
"""

import sys
import os
path = os.path.expanduser('~/') + 'imagezmq/imagezmq'
sys.path.insert(0, path)  # imagezmq.py /imagezmq
import socket
import time
import argparse
import cv2
from imutils.video import VideoStream
import imagezmq

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server-ip", required=True,
  help="ip address of the server to which the client will connect")
args = vars(ap.parse_args())
 
# initialize the ImageSender object with the socket address of the
# server
sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(
  args["server_ip"]))
 
# sender = imagezmq.ImageSender(connect_to='R-L-0000010:5555')
 
rpi_name = socket.gethostname() # send RPi hostname with each image
vs = VideoStream(src=0).start()
jpeg_quality = 95  # 0 to 100, higher is better quality, 95 is cv2 default
time.sleep(2.0)  # allow camera sensor to warm up


try:
  while True:  # send images as stream until Ctrl-C
    image = vs.read()
    # sender.send_image(rpi_name, image)
    ret_code, jpg_buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    sender.send_jpg(rpi_name, jpg_buffer)
    print('Image sent')

except KeyboardInterrupt:
  pass # Ctrl-C was pressed to end program; FPS stats computed below

except Exception as e:
  print('Python error: ', e)

finally:
  print('')
  print("Program terminated.")
  sys.exit()

















