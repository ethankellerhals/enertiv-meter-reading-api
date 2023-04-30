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
import torch.nn as nn

# self create cnn model for later use
#for loading the model
class analogReadingModel(nn.Module):
    def __init__(self):
        super(analogReadingModel, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=3,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(1280, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(12000, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization

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
imgs_cv2 = []
for image in s:
    if os.access(image, os.R_OK) == False:
        print(image +", image path is not valid")
        exit()
    imgs.append(Image.open(image))
    imgs_url.append(image)
    imgs_cv2.append(cv2.imread(image))

# image to tensor
transform = T.ToTensor()
for i, img in enumerate(imgs):

    img_t = transform(img)
    with torch.no_grad():
        prediction = model([img_t])
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
    crop_image = imgs_cv2[i][y:h, x:w]
    #get the type of the meter
    type = labels[index].item()
    value = 0
    confidence = 0
    meter_type = ""
    #anolog meter
    if type == 1:
        value,confidence = anolog.recognition(crop_image)
        meter_type = "anolog"
    #digital meter
    elif type == 2:
        value,confidence = digital.recognition(crop_image)
        meter_type = "digital"
    #dial meter
    elif type == 3:
        value,confidence = dial.recognition(crop_image)
        meter_type = "dial"
    
    print(imgs_url[i],"is", meter_type, "meter and the value is: ", value, " with confidence: ", confidence)

