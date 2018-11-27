import glob
import face_recognition
import numpy as np
import os

L = glob.glob("faces/*/*.jpg")
for l in L:
  e_file_name = l+"_encodings.txt"
  if os.path.isfile(e_file_name): continue
  I = face_recognition.load_image_file(l)
  E = face_recognition.face_encodings(I)
  if len(E)>0:
    np.savetxt(e_file_name,E[0])  
    print("[INFO] done {}".format(l)) 
  else:
    print("[WARNING] {} contains no face".format(l))
  
