import cv2 as cv
import numpy as np

# Distance constants 
KNOWN_DISTANCE = 190  # centimeters
PERSON_WIDTH = 46  # centimeters

# Object detector constant 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file 
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# set up OpenCV net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# object detector function/method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        if classid == 0:  # person class id
            label = "%s : %f" % (class_names[classid[0]], score)

            # draw rectangle on and label on object
            cv.rectangle(image, box, GREEN, 2)
            cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, GREEN, 2)
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1] - 2)])

    return data_list

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

ref_person = cv.imread('ReferenceImages/image6.png')

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

# finding focal length 
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

print(f"Person width in pixels : {person_width_in_rf} Focal length: {focal_person}")

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()

    data = object_detector(frame)
    for d in data:
        distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
        x, y = d[2]
        cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
        cv.putText(frame, f'Distance: {round(distance, 2)} cm', (x + 5, y + 13), FONTS, 0.48, GREEN, 2)

    cv.imshow('frame', frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cv.destroyAllWindows()
cap.release()