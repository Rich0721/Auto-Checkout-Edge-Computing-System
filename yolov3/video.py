from yolo_predict_paper import YOLO
from PIL import Image
import cv2
import numpy as np
from time import time
from glob import glob
import os


det =  YOLO(weight_path="./yolov3_copy/yolov3-136.h5", network='vgg')

#det = detector(weight_path="./tensorrt_files/ssd_mobilenetv1_float16.tflite")

def use_video(video_path=0):
    
    frame_number = 0
    
    cap = cv2.VideoCapture(video_path)
    _ = True
    t1 = time()
    while True:
        
        _, frame = cap.read()
        
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))
        frame = det.detect_image(frame)

        frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        
        frame_number+=1
        #frame = cv2.putText(frame, "FPS: {:.2f}".format(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #fps_list.append(fps)
        #cv2.imshow("video", frame)
        #cv2.waitKey(1)
    
    end = time() - t1
    fps = frame_number / end
    print("{}:{}".format(video_path, fps))
    return fps
    

def use_image(image_path):

    image = cv2.imread(image_path)
    predict_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predict_image = Image.fromarray(np.uint8(predict_image))
    results = det.detected_bbox(predict_image)

    for r in results:
        center_x = (r[0] + r[2]) // 2
        center_y = (r[1] + r[3]) // 2
        image = cv2.circle(image, (center_x, center_y), 10, (255, 255, 0), -1)
    cv2.imwrite(image_path[:-4] + "_result.jpg", image)

if __name__ == "__main__":
    fps = []
    videos = glob(os.path.join("./videos", "*.mp4"))
    for video in videos: 
        fps.append(use_video(video))
    fps = sorted(fps, reverse=True)
    print("{}".format(fps[4]))
    
    #use_image("./05685.jpg")