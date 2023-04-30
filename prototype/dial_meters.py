import sys
import cv2
import numpy as np
import pytesseract

CW_value_ranges = {
    0: [90, 54],
    1: [54, 18],
    2: [18, 342],
    3: [342, 306],
    4: [306, 270],
    5: [270, 234],
    6: [234, 198],
    7: [198, 162],
    8: [162, 126],
    9: [126, 90]
}

CCW_value_ranges = {
    0: [90, 126],
    1: [126, 162],
    2: [162, 198],
    3: [198, 234],
    4: [234, 270],
    5: [270, 306],
    6: [306, 342],
    7: [342, 18],
    8: [18, 54],
    9: [54, 90]
}

def Preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, gaussian_gray

def getValue(val: int, is_clockwise: bool) -> float:
    if is_clockwise:
        value = val
    else:
        value = 10. - val
    if value == 10:
        value = 0
    return value

def DialDetector(image, gaussian_gray):
    # find the circles
    circles = cv2.HoughCircles(gaussian_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=45, param1=150, param2=90, minRadius=0, maxRadius=0)
    # round all x, y, r values
    circles = np.uint16(np.around(circles))
    # Sort the circles from left to right (smallest to largest x-value)
    circles = circles[0][np.argsort(circles[:, :, 0])]
    # draw the circles
    for i in circles[0][:5]:
        # draws the outer part of the circle
        cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)
        # draws the center of the circle
        cv2.circle(image,(i[0],i[1]),2,(0,0,255),3)
    return circles


# inspiration for this function: 
# https://github.com/mirogta/dial-meter-reader-opencv-py
def getPointer(x: int, y: int, r: int, image) -> tuple:
    slices = 40 # = 36
    size = r * 0.8
    factor = 360 / slices
    pointer = None
    longest_dark = 0
    value = None
    center = tuple([x, y])
    for i in range(slices):
        angle = i * factor - 90
        dark_length = 0
        x2 = x + int(size * np.cos(angle * np.pi / 180))
        y2 = y + int(size * np.sin(angle* np.pi / 180))
        points_on_line = np.linspace(center, (x2 , y2), 255, 2)
        for p in points_on_line:
            point = np.int32(p)
            px = point[0]
            py = point[1]
            b = image[:, :, 0][py, px]
            g = image[:, :, 1][py, px]
            r = image[:, :, 2][py, px]
            gray = (b.astype(int) + g.astype(int) + r.astype(int)) / 3
            if gray < 130:
                dark_length += 1
            else:
                continue
        if dark_length > longest_dark:
            longest_dark = dark_length
            pointer = tuple(point)
            value = 10 * i / slices
    return value, pointer

# inspiration for this function: 
# https://github.com/mirogta/dial-meter-reader-opencv-py
def ProcessValues(values) -> str:
    meter_value = ""

    for i in reversed(values):
        value = int(np.floor(i))
        

    for i, (v) in enumerate(values):
        whole = int(np.floor(v))
        if i == len(values) - 1:
            meter_value += str(whole)
            break
        decimals = v - whole
        if decimals < 0.5 and values[i + 1] > 5:
            whole -= 1
        meter_value += str(whole)
    return meter_value

# GETTING THE DIAL ORIENTATION USING PYTESSERACT IS NOT 100% RELIABLE
def getDialOrientation(circles, image) -> bool:
    is_clockwise: bool
    circle = circles[0][0]
    x, y, r = circle
   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 9)
    mask = np.zeros_like(thresh)
    cv2.circle(mask, (x,y), r, 255, -1)
    thresh = cv2.bitwise_and(thresh, mask)

    x1 = int(x - r)
    x2 = int(x + r)
    y1 = int(y - r)
    y2 = int(y + r)

    cropped = thresh[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (150, 150))
 
    roi1 = resized[y1:y2, x1:x]
    roi2 = resized[y1:y2, x:x2]
    
    region1 = pytesseract.image_to_string(roi1, config='--psm 11 --oem 3 -c tessedit_char_whitelist=12346789')
    region2 = pytesseract.image_to_string(roi2, config='--psm 11 --oem 3 -c tessedit_char_whitelist=12346789')

    if any(digit in region1 for digit in ['1', '2', '3', '4']) or any(digit in region2 for digit in ['6', '7', '8', '9']):
        is_clockwise = False
    elif any(digit in region2 for digit in ['1', '2', '3', '4']) or any(digit in region1 for digit in ['6', '7', '8', '9']):
        is_clockwise = True
    else:
        return True
    return is_clockwise
    
def MeterValue(circles, image) -> str:
    values = []
    if getDialOrientation(circles, image):
        i = 0
    else:
        i = 1
    for c in circles[0]:
        
        x, y, r = c[0], c[1], c[2]

        val, tip = getPointer(x=x, y=y, r=r, image=image)
        angle = np.rad2deg(np.arctan2(-(tip[1]-y), tip[0]-x))
        if angle < 0:
            angle += 360.0
        cv2.putText(image, str(int(angle)), (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 255), 2)
        # draws line on the dial pointer
        cv2.line(image, (x, y), tip, (255, 0, 0), 2)

        if i % 2 == 0:
            #print(f"Dial {i}:", f"value = {getValue(val=val, is_clockwise=True)}")
            values.append(getValue(val=val, is_clockwise=True))
        else:
            #print(f"Dial {i}:", f"value = {getValue(val=val, is_clockwise=False)}")
            values.append(getValue(val=val, is_clockwise=False))
    
        i += 1
    
    return ProcessValues(values)


def main():
    img = cv2.imread(str(sys.argv[1]))
    gray, gaussian_gray = Preprocess(img)
    dials = DialDetector(img, gaussian_gray)
    print(f"meter value: {int(MeterValue(dials, img))} kWh")
    return 0

if __name__ == "__main__":
    main()