import torch
import cv2
import torchvision.transforms as T
import torchvision
import pytesseract
import numpy as np

def ocr(img):
    #image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #morphological operation
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))#13,5
    t = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    #thresholding
    ret,th1 = cv2.threshold(gray ,100,255,cv2.THRESH_BINARY_INV)
    #another thresholding
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
    #get the output first base on the grayscale image
    output = pytesseract.image_to_string(gray, lang='eng', config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
    c =  pytesseract.image_to_data(gray, output_type='data.frame', config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789',lang='eng')
    #if the output is empty, try to get the output from the thresholding image
    if(output == ''):
        thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        output = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
        c =  pytesseract.image_to_data(gray, output_type='data.frame', config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789',lang='eng')
    #if the output is empty, try to get the output from the thresholding image
    if(output == ''):
        output = pytesseract.image_to_string(th1, lang='eng', config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
        c =  pytesseract.image_to_data(gray, output_type='data.frame', config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789',lang='eng')
    #if the output is empty, try to get the output from the thresholding image
    if(output == ''):
        output = pytesseract.image_to_string(t, lang='eng', config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
        c =  pytesseract.image_to_data(gray, output_type='data.frame', config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789',lang='eng')
    #if the output is empty, try to get the output from the thresholding image
    if(output == ''):
        gray = cv2.medianBlur(th1, 3)
        ret,th1 = cv2.threshold(gray ,80,255,cv2.THRESH_BINARY_INV)
        ret,th1 = cv2.threshold(th1 ,100,255,cv2.THRESH_BINARY_INV)
        output = pytesseract.image_to_string(th1, lang='eng', config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
        c =  pytesseract.image_to_data(gray, output_type='data.frame', config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789',lang='eng')
    #get the confidence
    if c[c.conf != -1]['conf'].to_list() != []:
        conf =  c[c.conf != -1]['conf'].to_list()[0]
    #case cannot get any output
    else:
        conf = 0
    return output, conf
#convert the string to float
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
    file_boxes = []
    transform = T.ToTensor()
    #load digits model
    model = torch.load('digitsModel.pt')
    model.eval()
  
    #image to tensor
    img_t = transform(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    with torch.no_grad():
        prediction = model([img_t])
    #filter out low confidence boxes
    filtered_indices = sorted(
        torchvision.ops.nms(prediction[0]['boxes'], prediction[0]['scores'], iou_threshold=0.1).tolist(),
        key=lambda index: prediction[0]['boxes'][index, 0].item()
    )
    
    prediction[0] = {key: prediction[0][key][filtered_indices] for key in ('boxes', 'labels', 'scores')}

    bboxes, labels, scores= prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']
    #print(labels)
    boxes = bboxes.detach().numpy()
    l = labels.detach().numpy()
    #crop the image
    for i in range(len(boxes)):
        x,y,w,h = boxes[i]
        x,y,w,h = int(x), int(y), int(w), int(h)
        file_boxes.append(image[y:h, x:w])
    #ocr
    value_str = []
    s = 0
 
    for i in range(len(file_boxes)):
        output, conf = ocr(file_boxes[i])
        if output != '':
            value_str.append(output.strip('\n'))
        confidence += conf
        s += 1
   
    #check if the number of digits is correct
    value = label_check(value_str, s)


    return [value, confidence/s]