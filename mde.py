from roboflow import Roboflow
import cv2
import torch
import pandas as pd
import numpy
import matplotlib.pyplot as plt

# load the custom training YOLOv5 model
rf = Roboflow(api_key="LQk9XApDNPy9UBWO6j3l")
project = rf.workspace().project("miyo")
model = project.version(4).model

# load the pre-trained MiDaS model
midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
midas.to('cuda')
midas.eval()

transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.dpt_transform

# capture the input (0 for webcam)
cap = cv2.VideoCapture("input.mp4")

# get the width and height of captured frame
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# create an object to write the frames to output file
out = cv2.VideoWriter("output_test.mp4",cv2.VideoWriter_fourcc(*'MP4V'), 30, (frame_width,frame_height))

while cap.isOpened():
    ret, frame = cap.read()

    # YOLO
    output = model.predict(frame).json()

    # MiDaS
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cuda')

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
    for obj in output['predictions']:
        # if there is a prediction
        if output['predictions']:
            xmed = int(obj['x'])
            ymed = int(obj['y'])
        else:
            continue
        
        # calculating coordinates of bounding boxes by extracting x, y value information
        x0 = xmed - int(int(obj['width'])/2)
        y0 = ymed - int(int(obj['height'])/2)
        x1 = xmed + int(int(obj['width'])/2)
        y1 = ymed + int(int(obj['height'])/2)

        # extract label and confidence information
        label = str(obj['class'])
        conf = str(round(obj['confidence'], 2))

        print('x = ',xmed,'y = ',ymed)

        # colors for various classes
        if label == 'car':
            color = (0,255,0)
        elif label == 'pedestrian':
            color = (0,0,255)
        elif label == 'truck':
            color = (255,0,0)

        # calculating the distance measure
        inv = (pred[ymed][xmed])*0.001
        d = round(1/inv,2)
        
        # calculating actual distance value
        a = 0.12083143236283
        b = 3.78645050159133
        distance = str(round(a*d + b,1))

        # bounding box
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)

        # text box
        text_size, _ = cv2.getTextSize(label+' dist:'+distance, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        text_width, text_height = text_size
        cv2.rectangle(frame, (x0, y0), (x0+text_width, y0-text_height-10), color, -1)
        cv2.putText(frame, label+' dist:'+distance, (x0, y0-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
        
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