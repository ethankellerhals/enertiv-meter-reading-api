import torch
import torchvision
import cv2
import analogDetection as dd
from PIL import Image
from cnn_model import analog_cnn as cnn
import torch.nn.functional as F
    
# function to detect digits
def anologr(img, file_boxes, model):
    # image to tensor
    img_t,_= dd.get_transform(train=False)(Image.fromarray(img),Image.fromarray(img))
    with torch.no_grad():
        prediction = model([img_t])
    # filter out low confidence boxes
    filtered_indices = sorted(
        torchvision.ops.nms(prediction[0]['boxes'], prediction[0]['scores'], iou_threshold=0.5).tolist(),
        key=lambda index: prediction[0]['boxes'][index, 0].item()
    )
    prediction[0] = {key: prediction[0][key][filtered_indices] for key in ('boxes', 'labels', 'scores')}

    bboxes, labels, scores= prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']
    boxes = bboxes.detach().numpy()
    xywh = []
    #get the boxes of digits
    for i in range(len(boxes)):
        x,y,w,h = boxes[i]
        x,y,w,h = int(x), int(y), int(w), int(h)
        xywh.append([x,y,w,h])
        #cv2.rectangle(img, (x, y), (w, h), color = (255, 0, 0), thickness = 2)
    file_boxes.append(xywh)
    return file_boxes

# function to recognize digits
def get_value(img, model):
   
    img_tensor = cnn.train_transforms(Image.fromarray(img))
    o = model(img_tensor.unsqueeze(0))
    # Convert the logits to probabilities using softmax
    probs = F.softmax(o[0], dim=1)
    # Get the predicted class and the corresponding confidence level
    pred_class = torch.argmax(probs, dim=1)
    confidence_level = torch.max(probs, dim=1).values.item()
    #print('Predicted class: {}, confidence level: {}'.format(pred_class.item(), confidence_level))

    return [pred_class.item(), round(confidence_level,2)]

#convert the string to float, get the value of the meter
def label_check(labels, s):
    value = int(''.join(labels))
    if s == 8:
        value = float(value/1000)
    elif s == 7:
        value = float(value/100)
    elif s == 6:
        value = float(value/10)
    return value

def recognition(image):
    value = 0
    confidence = 0
    file_boxes = [] #store the boxes of digits in each image
    crop_imgs = [] #store the crop images of digits in each image
    val_conf = [] #store the string of digits in each image
    #load digits detection model
    model = torch.load('digitsModel.pt')
    model.eval()
    #get the boxes of digits
    anologr(image, file_boxes, model)
    #get the crop images of digits
    for i in range(len(file_boxes[0])):
        x,y,w,h = file_boxes[0][i]
        crop_imgs.append(image[y:h, x:w])
    #load digits recognition model
    model = torch.load('cnn_model/analogreadingmodel.pt')
    model.eval()
    #image to tensor and predict
    for i in crop_imgs:
        val_conf.append(get_value(i, model))
    # get value and confidence
    avg_conf = 0
    v = []
    for j in val_conf:
        v.append(str(j[0]))
        avg_conf += j[1]
    value = label_check(v,len(v))
    confidence  = round(avg_conf/len(v),2)

    return value, confidence