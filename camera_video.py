import cv2 as cv
import time
import geocoder
import os
import random

# Simulate accelerometer and gyroscope data
def get_simulated_accel_data():
    return {
        'x': random.uniform(-1, 1),
        'y': random.uniform(-1, 1),
        'z': random.uniform(-1, 1)
    }

def get_simulated_gyro_data():
    return {
        'x': random.uniform(-180, 180),
        'y': random.uniform(-180, 180),
        'z': random.uniform(-180, 180)
    }

# Define the video path and check its existence
video_path = "test.mp4"
absolute_path = os.path.abspath(video_path)
print(f"Checking file: {absolute_path}")

if not os.path.exists(absolute_path):
    print("File does not exist.")
    exit()

# Reading label names from obj.names file
class_name = []
with open(os.path.join("project_files", 'obj.names'), 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

# Importing model weights and config file
# Defining the model parameters
net1 = cv.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

# Defining the video source
cap = cv.VideoCapture(absolute_path)
if not cap.isOpened():
    print("Error: Video file could not be opened.")
    exit()

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Handling output directory and file
result_path = "pothole_coordinates"
os.makedirs(result_path, exist_ok=True)
output_file = os.path.join(result_path, 'result.avi')
result = cv.VideoWriter(output_file,
                         cv.VideoWriter_fourcc(*'MJPG'),
                         10, (width, height))

# Defining parameters for result saving and getting coordinates
try:
    g = geocoder.ip('me')
except Exception as e:
    print(f"Geocoder failed: {e}")
    g = None

starting_time = time.time()
Conf_threshold = 0.5
NMS_threshold = 0.4
frame_counter = 0
i = 0
b = 0

# Detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or cannot fetch the frame.")
        break

    frame_counter += 1

    # --- Simulate Sensor Data ---
    accel_data = get_simulated_accel_data()
    gyro_data = get_simulated_gyro_data()
    print(f"Simulated Accelerometer data: {accel_data}")
    print(f"Simulated Gyroscope data: {gyro_data}")

    # Log sensor data to a file
    with open(os.path.join(result_path, 'sensor_data.log'), 'a') as log_file:
        log_file.write(f"Frame {frame_counter}: Accel {accel_data}, Gyro {gyro_data}\n")
    # --- End of Simulation and Logging Code ---

    # Analyze the stream with detection model
    classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        label = "pothole"
        x, y, w, h = box
        recarea = w * h
        area = width * height

        # Drawing detection boxes on frame for detected potholes and saving coordinates txt and photo
        if len(scores) != 0 and score >= 0.7:
            if (recarea / area) <= 0.1 and y < 600:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame, f"{round(score * 100, 2)}% {label}",
                           (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

                if i == 0:
                    img_filename = os.path.join(result_path, f'pothole{i}.jpg')
                    cv.imwrite(img_filename, frame)
                    if g:
                        txt_filename = os.path.join(result_path, f'pothole{i}.txt')
                        with open(txt_filename, 'w') as f:
                            f.write(str(g.latlng))
                    i += 1
                else:
                    if (time.time() - b) >= 2:
                        img_filename = os.path.join(result_path, f'pothole{i}.jpg')
                        cv.imwrite(img_filename, frame)
                        if g:
                            txt_filename = os.path.join(result_path, f'pothole{i}.txt')
                            with open(txt_filename, 'w') as f:
                                f.write(str(g.latlng))
                        b = time.time()
                        i += 1

    # Writing FPS on frame
    ending_time = time.time() - starting_time
    fps = frame_counter / ending_time
    cv.putText(frame, f'FPS: {fps:.2f}', (20, 50), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    # Showing and saving result
    cv.imshow('frame', frame)
    result.write(frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        print("Quitting the video processing loop.")
        break

# Cleanup
cap.release()
result.release()
cv.destroyAllWindows()
