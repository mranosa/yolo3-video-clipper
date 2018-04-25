from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--video", dest = "videofile", help = "Video file to     run detection on", default = "video.avi", type = str)
    
    return parser.parse_args()
    
args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()



num_classes = 80
classes = load_classes("data/coco.names")

car_flag = -1
person_flag = -1
laptop_flag = -1
car_out = None
person_out = None
laptop_out = None

width = None
height = None

currentMilli = None

fourcc = cv2.VideoWriter_fourcc(*'MJPG')

#Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()


#Set the model in evaluation mode
model.eval()



def write(x, results, currentTime):
    global car_flag, person_flag, laptop_flag, car_out, person_out, laptop_out, width, height

    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    if label == "car":
        if car_flag == -1 :
            car_out = cv2.VideoWriter('output/' + str(int(currentMilli / 1000)) + 's - car.avi', fourcc, 20.0, (int(width),int(height)))
        car_flag = 30
        # cv2.putText(frame, 'has car:' + str(currentTime),
        #     (10, 40),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5, (128, 0, 0), 2)
    if label == "person":
        if person_flag == -1 :
            person_out = cv2.VideoWriter('output/' + str(int(currentMilli / 1000)) + 's - person.avi', fourcc, 20.0, (int(width),int(height)))
        person_flag = 30
    if label == "laptop":
        if laptop_flag == -1 :
            laptop_out = cv2.VideoWriter('output/' + str(int(currentMilli / 1000)) + 's - laptop.avi', fourcc, 20.0, (int(width),int(height)))
        laptop_flag = 30
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


videofile = args.videofile #or path to the video file. 

cap = cv2.VideoCapture(videofile)
fps = cap.get(cv2.CAP_PROP_FPS)

timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
calc_timestamps = [0.0]

#cap = cv2.VideoCapture(0)  for webcam

# Get current width of frame
width = cap.get(3)
height = cap.get(4)

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(width),int(height)))


assert cap.isOpened(), 'Cannot capture source'

frames = 0  
start = time.time()

while cap.isOpened():
    ret, frame = cap.read()    
    fps = cap.get(5)
    
    if car_out != None and car_flag > -1 :
        car_out.write(frame)
    if person_out != None and person_flag > -1 :
        person_out.write(frame)
    if laptop_out != None and laptop_flag > -1 :
        laptop_out.write(frame)

    if ret:
        # TODO find way to get the correct time in video
        # TODO clip the video into multiple files
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        currentMilli = cap.get(cv2.CAP_PROP_POS_MSEC)

        currentTime = calc_timestamps[-1] + 1000/fps
        calc_timestamps.append(currentTime)

        cv2.putText(frame, 'car flag:' + str(car_flag),
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (128, 0, 0), 2)

        img = prep_image(frame, inp_dim)
#        cv2.imshow("a", frame)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1,2)   
                     
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        output = model(Variable(img, volatile = True), CUDA)
        output = write_results(output, confidence, num_classes, nms_conf = nms_thesh)


        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
            print('car_flag : ' + str(car_flag))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue
        output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))

        im_dim = im_dim.repeat(output.size(0), 1)/inp_dim
        output[:,1:5] *= im_dim

        classes = load_classes('data/coco.names')
        colors = pkl.load(open("pallete", "rb"))

        list(map(lambda x: write(x, frame, currentMilli), output))
        
        cv2.imshow("frame", frame)
        # key = cv2.waitKey(int( (1 / int(fps)) * 1000))
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1
        print(time.time() - start)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
        print('car_flag : ' + str(car_flag))
        
        if car_flag > -1:
            car_flag -= 1
        if person_flag > -1:
            person_flag -= 1
        if laptop_flag > -1:
            laptop_flag -= 1
        if car_out != None and car_flag == 0:
            car_out.release()
        if car_out != None and person_flag == 0:
            person_out.release()
        if car_out != None and laptop_flag == 0:
            laptop_out.release()
    else:
        break     

# Release everything if job is finished
cap.release()
# out.release()

if car_out != None:
    car_out.release()
if person_out != None:
    person_out.release()
if laptop_out != None:
    laptop_out.release()            

cv2.destroyAllWindows()




