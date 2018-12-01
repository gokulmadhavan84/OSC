import numpy as np
import argparse
import cv2
import settings
import sys
import logging

_log = logging.getLogger(__name__)
out_hdl = logging.StreamHandler(sys.stdout)
out_hdl.setLevel(logging.DEBUG)
_log.addHandler(out_hdl)
_log.setLevel(logging.DEBUG)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
  "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
  "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
  "sofa", "train", "tvmonitor"]

class ObjectDetector(object):
  def __init__(self):
    _log.info("loading model...")
    self.net = cv2.dnn.readNetFromCaffe(
      settings.OBJ_PROTOTEXT_FILE,
      settings.OBJ_MODEL_FILE
    )

  def inference(self,image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
  
    # pass the blob through the network and obtain the detections and
    # predictions
    self.net.setInput(blob)
    detections = self.net.forward()
    # loop over the detections
    DETECTIONS = []
    for i in np.arange(0, detections.shape[2]):
      # extract the confidence (i.e., probability) associated with the
      # prediction
      confidence = detections[0, 0, i, 2]
      idx = int(detections[0, 0, i, 1])
  
      # filter out weak detections by ensuring the `confidence` is
      # greater than the minimum confidence
      if confidence > settings.OBJ_CUTOFF_THRESHOLD:
        # extract the index of the class label from the `detections`,
        # then compute the (x, y)-coordinates of the bounding box for
        # the object'
  
        box = detections[0, 0, i, 3:7] #* np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("float")
        
        DETECTIONS.append({
          'x1':float(startX),
          'y1':float(startY),
          'x2':float(endX),
          'y2':float(endX),
          'c' :float(confidence),
          'l' :CLASSES[idx]
        })
    return DETECTIONS

  def vis(self,frame, DETECTIONS):
    image = frame.copy()  
    for detection in DETECTIONS:
      x1,y1,x2,y2,c,l = detection['x1'], detection['y1'], detection['x2'], detection['y2'], detection['c' ], detection['l' ]
      label = "{}: {:.2f}%".format(l, c)
      color = detection['color']
      cv2.rectangle(image, (x1, y1), (x2,y2), color, 2)
      y = y1 - 15 if y1 - 15 > 15 else y1 + 15
      cv2.putText(image, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, detection['color'], 2)
    return image

