import cv2
import numpy as np


# --------- EVALUATION ----------

# CW = clockwise, CCW = counterclockwise
# 4-dial meter dial orientation = (CCW, CW, CCW, CW)
# 5-dial meter dial orientation = (CW, CCW, CW, CCW, CW)

#Three metrics for evaluation:
# (1) meter recognition rate (MRR)
# (2) dial recognition rate (DRR)
# (3) mean absolute error (MAE)


image = cv2.imread("path/..")

boxes = model(image) # get the bounding boxes from the Faster R-CNN model

# crop the images of the dials
dial_images = []
for box in boxes:
    x1, y1, x2, y2 = box.tolist()

    dial_image = image[y1:y2, x1:x2]
    dial_images.append(dial_image)


for i, dial_image in enumerate(dial_images):
    gray = cv2.cvtColor(dial_image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPOX_SIMPLE)

    box = boxes[i]
    x1, y1, x2, y2 = box.tolist()
    for contour in contours:
        shifted_contour = contour + np.array([x1,y1])
        cv2.drawContours(image, [shifted_contour], 0, (0, 255, 0), 2)



cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()







# Object detection with recognition

# Object detection with regression

# - AngReg

# Segmentation-free recognition

# Data augmentation

# -need library 'Albumentations' for image augmentation

# Post-correction



