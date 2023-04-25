import torch
from PIL import Image
import cv2
import numpy as np
import detection as det
import anologRecognition as anolog
import digitalRecognition as digital
import dialRecognition as dial
import sys
import os
import torchvision
import torchvision.transforms as T

#load model
model = torch.load('detectionModel.pt')
model.eval()

# image check
s = sys.argv[1:]
if len(s) == 0:
    print("Please provide the path to the image")
    exit()

imgs = []
imgs_url = []
for image in s:
    if os.access(image, os.R_OK) == False:
        print(image +", image path is not valid")
        exit()
    imgs.append(Image.open(image))
    imgs_url.append(image)

# image to tensor
transform = T.ToTensor()
for i, img in enumerate(imgs):

    img = transform(img)
    with torch.no_grad():
        prediction = model([img])
    # filter out low confidence boxes
    filtered_indices = sorted(
        torchvision.ops.nms(prediction[0]['boxes'], prediction[0]['scores'], iou_threshold=0.1).tolist(),
        key=lambda index: prediction[0]['boxes'][index, 0].item()
    )
    prediction[0] = {key: prediction[0][key][filtered_indices] for key in ('boxes', 'labels', 'scores')}
    bboxes, labels, scores= prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']
    #get the crop image
    index = torch.argmax(scores).item()
    bbox = bboxes[index]
    x,y,w,h = bbox.detach().numpy()
    x,y,w,h = int(x), int(y), int(w), int(h)
    crop_image = img[y:h, x:w]
    #get the type of the meter
    type = labels[index].item()
    value = 0
    confidence = 0
    #anolog meter
    if type == 1:
        value,confidence = anolog.recognition(crop_image)
    #digital meter
    elif type == 2:
        value,confidence = digital.recognition(crop_image)
    #dial meter
    elif type == 3:
        value,confidence = dial.recognition(crop_image)
    
    print("The value of ",imgs_url[i]," is: ", value, " with confidence: ", confidence)