import os
import pyblake2
import glob
import numpy as np
from PIL import Image
import cv2
import subprocess
import io

class VideoObject(object):
  def __init__(self, video_file, cache_dir=None):
    self.file_path = video_file.strip()
    self._computed_length=None
    if not cache_dir:
      d,p = os.path.split(self.file_path)
      p = ''.join(p.split('.'))
      cache_dir = os.path.join(d,p)
    
    self.cache_dir = cache_dir
    os.system("mkdir -p {}".format(self.cache_dir))    
    os.system("mkdir -p {}/{}".format(self.cache_dir,"meta"))    
  
  def get_meta_path(self, time_ms, algo):    
    time_stamp = self._get_timestamp(time_ms)

    path = self.cache_dir+"/meta/frame_{}_{}.json".format(time_stamp,algo)   
    print(path)
    return path
 
  def get_frames(self, resolution_ms=40):
    '''
    Get frames of a video one after the other as a list(generator)
    '''
    for time_ms in range(0,self.length, resolution_ms):
      frame_obj = self.get_frame(time_ms)
      yield frame_obj

  def _execute_system_command(self, cmd_list):
    result = subprocess.run(cmd_list, stdout=subprocess.PIPE)
    assert result.returncode==0, result
    return result.stdout

   
  def _get_timestamp(self,tot_ms):
    t = tot_ms
    t,ms = divmod(t,1000)
    t,sc = divmod(t,60)
    t,mn = divmod(t,60)
    t,hr = divmod(t,60)
    return "{:02d}:{:02d}:{:02d}.{:03d}".format(hr,mn,sc,ms)

  def _get_milliseconds(self, time_stamp):
    time_stamp = str(time_stamp)
    hh,mm,ss_t = time_stamp.split(":")
    ss,us = ss_t.split(".")
    hh,mm,ss,us = [int(t) for t in [hh,mm,ss,us]]
    return (hh*60*60*1000)+(mm*60*1000)+(ss*1000)+(us//1000)
  
  def get_frame(self, time_ms):
    '''
    Get a frame of a video 
    '''
    cmd = [
      "ffmpeg",
      "-ss","{offset_time}".format(
        offset_time=self._get_timestamp(time_ms),
      ),                    # start offset
      "-loglevel", "panic", # be quiet
      "-y",                 # dont ask for confirmation
      "-i","{}".format(self.file_path),    # input file
      "-vframes", "1",      # no of frames to extract
      "-f","image2pipe",    # op but print out
      "-vcodec","{fmt}".format(fmt="bmp"), # image encoing format
      "pipe:1",             # write to stdout
    ]

    img_raw = self._execute_system_command(cmd)
    frame = Image.open(io.BytesIO(img_raw)).convert("RGB")
    return frame

  @property
  def width(self): 
    I_obj = self.get_frame(40) 
    width = I_obj.width
    return width

  @property
  def height(self): 
    I_obj = self.get_frame(40) 
    height=I_obj.height
    return height

  @property
  def length(self):
    '''
      returns length in milliseconds
    '''
    length_cmd = [
      "ffprobe",
      "-v",
      "error",
      "-show_entries",
      "format=duration",
      "-of",
      "default=noprint_wrappers=1:nokey=1",
      "-sexagesimal",
      self.file_path
    ]
    result = self._execute_system_command(length_cmd)
    result = result.decode('utf-8').strip()
    return self._get_milliseconds(result)
    
  def __len__(self):
    return self.length
