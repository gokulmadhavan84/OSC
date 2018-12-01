import time
from threading import Thread
from video import VideoObject #, VideoWriter
#import requests
from PIL import Image
from io import BytesIO
import base64
import json,os
import glob
import numpy as np
import settings
import cv2
#from bbox_visualiser import draw_bboxes_on_image
#import concurrent.futures
from object_detector import ObjectDetector

_OBD = ObjectDetector()

def send_request_object(frame):
  frame = np.asarray(frame)
  #buff = BytesIO()
  #Image.fromarray(frame).save(buff, format='png')
  #files = {'image': buff.getvalue()}

  #r=requests.post(settings.OBJECT_URL,files=files)
  D = _OBD.inference(frame)
  print("[OBJECT] Done")
  return D

def get_req_handled(method, frame, cache_file): 
  if not os.path.isfile(cache_file):
    data = method(frame)
    with open(cache_file,'w') as f:
      json.dump(data, f, indent=2)
  else:
    with open(cache_file,'r') as f:
      data = json.load(f)
  return data

def one_thread(V,time_ms):
  # what does one thread has to do
  result_file = V.get_meta_path(time_ms,'obj')
  frame = V.get_frame(time_ms)
  meth = send_request_object
  result = get_req_handled(meth, frame, result_file)
  return result


def compute_detection_results(video_path): 
  V = VideoObject(video_path)
  video_length = V.length
  all_frames_time_to_process = list(range(0,video_length, settings.REQUEST_RESOLUTION)) 
  
  frames_time_to_process = []
  for f_no in all_frames_time_to_process:
    result_file = V.get_meta_path(f_no,'obj')
    if not os.path.isfile(result_file):
      frames_time_to_process.append(f_no)

  print("[INFO] total frames to process",len(all_frames_time_to_process))
  print("[INFO] actual frames to process",len(frames_time_to_process))  

  for f_ms in frames_time_to_process:
      #try:
      one_thread(V,f_ms)
      #except Exception as exc:
      #print('Exception at [{:d}]: {}'.format(f_no, exc))

#os.system("mkdir -p {}".format(settings.OUTPUT_FOLDER))

video_list = list(sorted(glob.glob(settings.INPUT_FOLDER+"/*.*")))
#"""
for path in video_list:
  print("[INFO] computing ",path)
  compute_detection_results(path) 
