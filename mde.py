import sys
sys.path.append('C:\\Users\\sahar\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python310\\site-packages\\')

from roboflow import Roboflow
import cv2
import torch
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import math
import yaml
import time
import json
from LinearRegressionModel import LinearRegressionModel
from sort import *


if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 
device = torch.device(dev)

model = torch.hub.load('ultralytics/yolov5','custom','best.pt',verbose=False,device=dev)

# load the pre-trained MiDaS model
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small',verbose=False,device=dev)
midas.to(device)
midas.eval()

transforms = torch.hub.load('intel-isl/MiDaS', 'transforms',verbose=False)
transform = transforms.small_transform

# load the regression model
linear = LinearRegressionModel()
try:
    state_dict = torch.load("C:\\Users\HP\\Desktop\\monocular_depth_estimation\\final\\one_new.pt")
except Exception as e:
    print(f"An error occurred while loading the state dictionary: {e}")

# Set the state dictionary to the linear regression model
linear.load_state_dict(state_dict)

# Set the model to evaluation mode
linear.eval()

# capture the input (0 for webcam)
cap = cv2.VideoCapture("input_480p.mp4")

# get the width and height of captured frame
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# create an object to write the frames to output file
out = cv2.VideoWriter("output_480p.mp4",cv2.VideoWriter_fourcc(*'MP4V'), 60, (frame_width,frame_height))

# accessing configuration file
with open('config.yml') as file:
    list = yaml.load(file, Loader=yaml.FullLoader)
print(list)
count = 0

mot_tracker = Sort()

while cap.isOpened():
    # start counter
    start = time.perf_counter()

    ret, frame = cap.read()

    # YOLO
    result = model(frame)
    output = result.pandas().xyxy[0]

    detections = result.pred[0].numpy() #
    track_bbs_ids = mot_tracker.update(detections) #

    # MiDaS
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2],
            mode = "bicubic",
            align_corners = False,
        ).squeeze()

        pred = prediction.cpu().numpy()

    # YOLO
    for obj in output.iterrows():
        # if there is a prediction
        if obj[1]['name']:
            # calculating coordinates of bounding boxes by extracting x, y value information
            x0 = int(obj[1]['xmin'])
            y0 = int(obj[1]['ymin'])
            x1 = int(obj[1]['xmax'])
            y1 = int(obj[1]['ymax'])
        else:
            continue
        
        min = 10000
        for j in range(len(track_bbs_ids.tolist())): #
            coords = track_bbs_ids.tolist()[j]
            x2,y2,x3,y3 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]),
            id_x = int(coords[4])

            diff = abs(x0-x2)+abs(y0-y2)+abs(x1-x3)+abs(y1-y3)
            if diff<min:
                min = diff
                id = str(id_x)

        xmed = int((x1 + x0)/2)
        ymed = int((y1 + y0)/2)

        # extract label and confidence information
        label = str(obj[1]['name'])
        conf = str(round(obj[1]['confidence'], 2))
    
        # colors for various classes
        if label in list['Color'] and obj[1]['confidence']>=list['Threshold']['Confidence']:
            color=eval(list['Color'][label])

            # calculating the distance measure
            inv = (pred[ymed][xmed])*0.001
            inv = round(1/inv,2)
            X = [inv,xmed,ymed]
            Val = torch.Tensor(X)
            with torch.inference_mode():
                linear_pred = linear(Val)   
            distance_list = linear_pred.tolist()
            distance = distance_list[0]
            distance = round(distance,1)
        
            distance = str(distance)

            # bounding box
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)

            # text box
            text_size, _ = cv2.getTextSize(label+' '+distance+'m ', cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_size2, _ = cv2.getTextSize('ID: '+id, cv2.FONT_HERSHEY_PLAIN, 1, 1)

            text_width, text_height = text_size
            text_width2, text_height2 = text_size2
            cv2.rectangle(frame, (x0, y0), (x0+text_width, y0-text_height-5), color, -1)
            cv2.putText(frame, label+' '+distance+'m ', (x0, y0-5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x1-text_width2, y1), (x1, y1-text_height2-5), color, -1)
            cv2.putText(frame, 'ID: '+id, (x1-text_width2, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)

    end = time.perf_counter()

    totaltime = end - start
    fps = 1 / totaltime
    cv2.putText(frame, 'FPS: '+str(int(fps)), (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    
    # display the frame and save it to file
    cv2.imshow("Depth Estimation",frame)
    out.write(frame)

    # close window on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

# release capture and close all windows
cap.release()
cv2.destroyAllWindows()