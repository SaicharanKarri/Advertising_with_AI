from ultralytics import YOLO
import cv2
import numpy as np
import os
from copy import deepcopy
import tensorflow
from tensorflow import keras
import json



class pipeline:

    # class attributess
    
    object_detection_model_path = r"weights\object_detection\yolov8n.pt"
    image_classification_model_path = r"weights\image_classification\mobnet_v2.tflite"
    output_file = r"output/output.avi"

    # instance attributes

    def __init__(self,video_path,advertisements_folder_path,json_path):
        self.video_path = video_path
        self.advertisements_folder_path = advertisements_folder_path
        self.json_path = json_path

    def load_json(self):
        with open(self.json_path,"r") as file:
            var = json.load(file)
            file.close()
        return var

        
    # method for loading object detection and image classification model

    def load_models(self):
        
        obj_model = YOLO(self.object_detection_model_path)
        interpreter = tensorflow.lite.Interpreter(model_path=self.image_classification_model_path)
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]
        return obj_model,interpreter,input_index,output_index

    # method for loading the video
    def load_video(self,obj_det_model,img_cls_model,input_index,output_index,roi):
        source = cv2.VideoCapture(self.video_path)
        frame_width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = source.get(cv2.CAP_PROP_FPS)
        frame_size = (frame_width, frame_height)

        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        out = cv2.VideoWriter(self.output_file, fourcc, fps, frame_size)

        roi = np.array(self.load_json()["points"],dtype=np.int32)
        roi = roi.reshape((-1, 1, 2))
        adverstisements = self.load_advertisements()

        while True:
            ret,frame = source.read()
            # female_frame = deepcopy(frame)
            # male_frame = deepcopy(frame)
            if ret:
                results = obj_det_model.predict(frame,classes=[0])
                # if there is atleast one detection in the frame
                if len(results[0]) >=1:
                    # count_of_objects = results[0].boxes.xyxy.numpy()
                    count_male = 0
                    count_female = 0
                    for coordinates in results[0].boxes.xyxy.numpy():
                        x1 = int(coordinates[0])
                        y1 = int(coordinates[1])
                        x2 = int(coordinates[2])
                        y2 = int(coordinates[3])
                        
                        person = np.expand_dims(cv2.resize(frame[y1:y2,x1:x2],(224,224)), axis=0).astype(np.float32)
                        center = (int((x1+x2)/2),int((y1+y2)/2))
                        
                        if cv2.pointPolygonTest(roi,center,False) >=1:
                            img_cls_model.set_tensor(input_index, person)
                            img_cls_model.invoke()
                            predictions = img_cls_model.get_tensor(output_index)[0][0] 
                            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                            
                            if predictions >=0.5:
                                cv2.putText(frame,f"male", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1)
                                count_male +=1
                            else:
                                cv2.putText(frame,f"female", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1)
                                count_female +=1

                if count_male < count_female:
                    frame[0:300,1000:1280] = adverstisements["file_1"]
                elif count_male == count_female:
                    frame[0:300,1000:1280] = adverstisements["file_0"]
                else:
                    frame[0:300,1000:1280] = adverstisements["file_2"]
                cv2.putText(frame,f"count of male = {count_male}", (30, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(frame,f"count of female = {count_female}", (30, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.polylines(frame, [roi], True, (255, 0, 0), 2)
                out.write(frame)
                cv2.imshow("window",frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        out.write(frame)
        cv2.destroyAllWindows()
        out.release()
        source.release()

    def load_advertisements(self):
        db = {}
        c = 0
        for file in os.listdir(self.advertisements_folder_path):
             img = cv2.imread(self.advertisements_folder_path + "/" + file)
             img = cv2.resize(img,(280,300))
             db[f"file_{c}"] = img
             c+=1
        return db # dictionary {file1:nparray,file2:nparray}


