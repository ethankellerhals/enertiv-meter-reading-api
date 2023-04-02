import os
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from functools import lru_cache
import pandas as pd
from PIL import Image
import cv2
import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'detection/')

from engine import train_one_epoch, evaluate
import utils
import transforms as T

#helper function to get the contours
def get_contour(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds of red color in HSV color space
    lower_red = np.array([0, 150, 150])
    upper_red = np.array([15, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 150, 150])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Combine masks
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x = []
    y = []
    w = []
    h = []
    #find the bounding boxes of the contour
    if len(contours) > 0:
        for i in range(len(contours)):
            x1, y1, w1, h1 = cv2.boundingRect(contours[i])
            if cv2.contourArea(contours[i]) > 200:
                x.append(x1)
                y.append(y1)
                w.append(w1)
                h.append(h1)

    return x,y,w,h

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class MeterReadingDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "annotate_crop"))))  # this will be the orginal image
        self.masks = list(sorted(os.listdir(os.path.join(root, "annotate_boxes")))) # this will be the annotated image
        self.values = pd.read_csv(os.path.join(root, "annotate.csv"), converters={'value': str})

    def __getitem__(self, idx):
        # load images and masks
        #print("let me see:", self.imgs[idx], idx)
        img_path = os.path.join(self.root, "annotate_crop", self.imgs[idx])
        mask_path = os.path.join(self.root, "annotate_boxes", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        boxes = []
        anotate_img = cv2.imread(mask_path)
        #print(self.masks[idx])
        x,y,w,h = get_contour(np.array(anotate_img))
        for i in range(len(x)):
            xmax = x[i] + w[i]
            ymax = y[i] + h[i]
            xmin = x[i]
            ymin = y[i]
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # the value of the meter reading
        value_array = self.values[self.values['fileid'] == self.imgs[idx]]['value'].values[0]
        value_array = [*value_array]
        value_array = list(map(int, value_array))
        num_digits = len(value_array)
        labels = torch.zeros(num_digits, dtype=torch.int64)
        for digit_index, digit in enumerate(value_array):
            labels[digit_index] = digit+1
        if len(boxes) != num_digits:
            print(self.imgs[idx],mask_path, len(boxes), labels)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def load_model():
    num_classes = 11 # 10 ditgits and background
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_dataloader():
    # use our dataset and defined transformations
    dataset = MeterReadingDataset('../data/annotation_v2', get_transform(train=True))
    dataset_test = MeterReadingDataset('../data/annotation_v2', get_transform(train=False))
    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[-50:])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
  
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    return data_loader, data_loader_test

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #load data
    data_loader, data_loader_test = get_dataloader()
 
    # get the model using our helper function
    model = load_model()

    # move model to the right device
    model.to(device)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 20
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=20)
        # update the learning rate
        lr_scheduler.step()
        # # evaluate on the test dataset
        #evaluate(model, data_loader_test, device=device)
    
    torch.save(model, "model.pt")
    print("saved")

if __name__ == "__main__":
    main()