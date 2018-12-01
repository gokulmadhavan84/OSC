import os

DIR = os.path.dirname(os.path.realpath(__file__))


OBJ_CUTOFF_THRESHOLD = 0.8

OBJ_PROTOTEXT_FILE=os.path.join(DIR,"MobileNetSSD_deploy.prototxt.txt")
OBJ_MODEL_FILE=    os.path.join(DIR,"MobileNetSSD_deploy.caffemodel")
INPUT_FOLDER=    os.path.join(DIR,"input_files")
REQUEST_RESOLUTION=250
