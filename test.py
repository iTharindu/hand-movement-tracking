import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json


with open("config.json") as json_config_file:
    config = json.load(json_config_file)

video_path = config["video_path"]

img_array = []
count = 1
cap = cv2.VideoCapture(video_path)
while(cap.isOpened()):                    # play the video by reading frame by frame
    ret, frame = cap.read()
    if ret==False:
        print("Video ended")
        break
    print(count)
    count += 1
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
print("done")
