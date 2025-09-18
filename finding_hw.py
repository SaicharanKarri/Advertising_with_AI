from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\saich\Downloads\best.pt")

width = []
height = []


source = cv2.VideoCapture(r"C:\Users\saich\OneDrive\Desktop\Smart Advertising System\videos\floor.mp4")

while True:
    ret,frame = source.read()

    if ret:

        results = model.predict(frame)
        if len(results[0]) >=1:
            for coordinates in results[0].boxes.xywh.numpy():
                x = int(coordinates[0])
                y = int(coordinates[1])
                w = int(coordinates[2])
                h = int(coordinates[3])
                width.append(w)
                height.append(h)
    else:
        break

print("average width of all the crops is",sum(width)/len(width))
print("average height of all the crops is",sum(height)/len(height))