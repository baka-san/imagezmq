import sys
import os
zmq_path = os.path.expanduser('~/') + 'imagezmq/imagezmq'
sys.path.insert(0, zmq_path)  # imagezmq.py /imagezmq
import time
import darknet as dn
import numpy as np
import cv2
import imagezmq
from PIL import Image, ImageFile

ImageFile.MAXBLOCK = 2**20
image_hub = imagezmq.ImageHub()
total_time = 0
frame_count = 0
dn.set_gpu(0)
net = dn.load_net(b'cfg/yolov3-safety-vest.cfg', b'backup/yolov3-safety-vest_9000.weights', 0)
meta = dn.load_meta(b'safety-vest.data')
results_path = os.path.expanduser('~/') + 'yolo/darknet/results/frames/'
print('Neural net loaded. Ready for frames.')

def drawBoundingBoxes(detections, image):
  # Initialize some variables
  result = {
    'image': image
  }
  label = 'Nothing detected'

  try:
    for detection in detections:
      objectClass = detection[0].decode("utf-8") 
      confidence = detection[1]
      label = objectClass + ': ' + str(np.rint(100 * confidence)) + '%'

      # x-center, y-center, x-width, y-width
      bounds = detection[2]

      # Set the bounding coords
      x1 = int(bounds[0]) - int(bounds[2]/2)
      y1 = int(bounds[1]) - int(bounds[3]/2)
      x2 = int(bounds[0]) + int(bounds[2]/2)
      y2 = int(bounds[1]) + int(bounds[3]/2)

      # Draw the bounding box
      cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 5)

      # Write a label
      cv2.putText(image, label, (x1+5, y1+40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2) 

    result = {
      'detections': detections,
      'image': image,
      'caption': '\n<br/>'.join(label)
    }
  except Exception as e:
    print("Unable to draw boxes: ", e)

  return result


try:
  while True:  # show streamed images until Ctrl-C
    # Receive frame
    rpi_name, jpg_buffer = image_hub.recv_jpg()

    # Set timer to track FPS
    if frame_count > 0:
      start_time = time.time()

    # Decode the image
    image = cv2.imdecode(np.fromstring(jpg_buffer, dtype='uint8'), -1)

    # Save image resolution
    if frame_count == 0:
      height, width = image.shape[:2]

    # Detect objects
    detections = dn.detect(net, meta, image, thresh=0.5, hier_thresh=0.5)
    print('detections = ', detections)

    # Draw bounding box on image
    result = drawBoundingBoxes(detections, image)

    # Save image
    try:
      file_path = results_path + 'frame-' + str(frame_count) + '.jpg'
      print('Filename: ', file_path)
      cv2.imwrite(file_path, result['image'])

    except Exception as e: 
      print("Couldn't save file: ", e) 

    # Measure processing time
    if frame_count > 0:
      processing_time = time.time() - start_time
      total_time += processing_time
      print('Processing time: ', processing_time)
    
    frame_count += 1

    # Ask client for another frame
    image_hub.send_reply(b'OK')

except KeyboardInterrupt:
  pass # Ctrl-C was pressed to end program; FPS stats computed below

except Exception as e:
  print('Python error: ', e)

finally:
  print('')
  print('=========== SUMMARY ===========')
  print('Results: ', results_path)
  print('Total images: {:,g}'.format(frame_count))
  if frame_count == 0:
    sys.exit()
  print('Stream resolution: {}x{}'.format(width, height))
  fps = frame_count/total_time
  print('Approximate FPS: ', fps)
  sys.exit()

# def drawBoundingBoxes(detections, image):
#     try:
#         from skimage import io, draw
#         import numpy as np
#         print("*** "+str(len(detections))+" Results, color coded by confidence ***")
#         imcaption = []
#         for detection in detections:
#             label = detection[0].decode()
#             confidence = detection[1]
#             pstring = label+": "+str(np.rint(100 * confidence))+"%"
#             imcaption.append(pstring)
#             print(pstring)
#             bounds = detection[2]
#             shape = image.shape
#             yExtent = int(bounds[3])
#             xEntent = int(bounds[2])
#             # Coordinates are around the center
#             xCoord = int(bounds[0] - bounds[2]/2)
#             yCoord = int(bounds[1] - bounds[3]/2)
#             boundingBox = [
#                 [xCoord, yCoord],
#                 [xCoord, yCoord + yExtent],
#                 [xCoord + xEntent, yCoord + yExtent],
#                 [xCoord + xEntent, yCoord]
#             ]
#             # Wiggle it around to make a 3px border
#             rr, cc = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
#             rr2, cc2 = draw.polygon_perimeter([x[1] + 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
#             rr3, cc3 = draw.polygon_perimeter([x[1] - 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
#             rr4, cc4 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] + 1 for x in boundingBox], shape= shape)
#             rr5, cc5 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] - 1 for x in boundingBox], shape= shape)
#             boxColor = (int(255 * (1 - (confidence ** 2))), int(255 * (confidence ** 2)), 0)
#             draw.set_color(image, (rr, cc), boxColor, alpha= 0.8)
#             draw.set_color(image, (rr2, cc2), boxColor, alpha= 0.8)
#             draw.set_color(image, (rr3, cc3), boxColor, alpha= 0.8)
#             draw.set_color(image, (rr4, cc4), boxColor, alpha= 0.8)
#             draw.set_color(image, (rr5, cc5), boxColor, alpha= 0.8)
#         detections = {
#             "detections": detections,
#             "image": image,
#             "caption": "\n<br/>".join(imcaption)
#         }
#     except Exception as e:
#         print("Unable to draw boxes: "+str(e))
#     return detections
