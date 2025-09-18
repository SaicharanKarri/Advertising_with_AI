import cv2
import json
import argparse

data = {"points":[]}
def on_click(event,x,y,flags,paramters):
    if event == cv2.EVENT_LBUTTONDOWN:
        data["points"].append([x,y])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path",type=str,required=True)
    args = parser.parse_args()
    video_path = args.video_path
    source = cv2.VideoCapture(video_path)

    # accessing the frame
    c = 0
    while True:
        ret,frame = source.read()
        if ret:
            c += 1
        if c == 10:
            break
        else:
            break

    # creating roi on the image
    cv2.namedWindow("ROI")
    while True:
        cv2.setMouseCallback("ROI",on_click)
        cv2.imshow("ROI",frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        cv2.destroyAllWindows()

    with open("annotations.json","w") as file:
        json.dump(data,file)
        file.close()




    