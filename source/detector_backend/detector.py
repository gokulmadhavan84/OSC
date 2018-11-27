import cv2
import face_recognition
from PIL import Image
import glob
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from flask import Flask, render_template, Response
from camera import VideoCamera

app = Flask(__name__)


ENCODINGS = []
NAMES = []
_font = ImageFont.truetype("arial.ttf", 18)

for l in glob.glob("faces/*/*encodings.txt"):
  enc = np.loadtxt(l)
  name = l.split('/')[-2]
  ENCODINGS.append(enc)
  NAMES.append(name)

def face_detection(image):
  delta = 2
  small_image = cv2.resize(image, (0, 0), fx=1.0/delta, fy=1.0/delta)
  face_locations = face_recognition.face_locations(small_image, model="hog")
  face_locations = [ (t*delta,r*delta,b*delta,l*delta) for t,r,b,l in face_locations]
  return face_locations

def visualise(image,face_locations, face_encodings):
  I = Image.fromarray(np.uint8(image))
  draw = ImageDraw.Draw(I)

  face_names = []
  for face_encoding in face_encodings:
    face_distances = face_recognition.face_distance(ENCODINGS, face_encoding)
    idx = np.argmin(face_distances)
    if face_distances[idx] < 0.37:
      #name = "{} {:4.2f}".format(NAMES[idx],face_distances[idx])
      name = NAMES[idx].split("_")[0]
    else:
      name = "" # "unknown"
    face_names.append(name)

  for (top, right, bottom, left), name in zip(face_locations, face_names):
    text_size = _font.getsize(name)
    draw.rectangle(((left, top), (right, bottom)),outline="#00ff00")
    draw.rectangle(((left+1, top+1), (left+text_size[0]+1,top+text_size[1]+1)),fill="#00ff00")
    draw.text((left+1,top+1), name, font=_font, fill="#000000")

  image = np.array(I)
  return image

def process(image):
  face_locations = face_detection(image)
  face_encodings = face_recognition.face_encodings(image, face_locations)
  image = visualise(image, face_locations, face_encodings) 
  return image

class VideoCamera(object):
  def __init__(self):
    self.video = cv2.VideoCapture(0)
    
  def __del__(self):
    self.video.release()
    
  def get_frame(self):
    success, frame = self.video.read()
    frame = process(frame) 
    ret, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()

def gen(camera):
  while True:
    frame = camera.get_frame()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
  return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=False)
