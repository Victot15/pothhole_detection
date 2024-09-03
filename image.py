import cv2 as cv
import numpy as np
import os

# Ensure image path is correct
image_path = 'img1.jpg'
absolute_path = os.path.abspath(image_path)
print(f"Checking file: {absolute_path}")

if not os.path.exists(absolute_path):
    print("File does not exist.")
    exit()

# Read the image
img = cv.imread(absolute_path)
if img is None:
    print("Error: Unable to load image.")
    exit()

# Import model weights and config file
# Defining the model parameters
net = cv.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

# Perform detection
try:
    classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)
    for (classid, score, box) in zip(classIds, scores, boxes):
        x, y, w, h = box
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(img, f'{score:.2f}', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the result
    cv.imshow('Detection', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

except cv.error as e:
    print(f"OpenCV error: {e}")

except Exception as e:
    print(f"An error occurred: {e}")
