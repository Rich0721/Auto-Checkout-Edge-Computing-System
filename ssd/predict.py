
from ssd_predict import detector
from PIL import Image
import cv2
import tensorflow as tf
import numpy as np



predicts = []
with open("test.txt") as f:
    line = f.readline()
    while line:
        l = line.split()
        
        #boxes = np.array([np.array(list(map(int, box.split(",")))) for box in l[1:]])
        
            
        predicts.append([l[0]])

        line = f.readline()
#weight_paths = ["./ssd_version/ssd_mobilenetv1.h5","./ssd_version/ssd_mobilenetv2.h5", "./ssd_version/ssd_mobilenetv3.h5", 
#                "./ssd_version/ssd_shufflenetv1.h5", "./ssd_version/ssd_shufflenetv2.h5", "./ssd_version/ssd_shuffle_mobilenet.h5"]

#networks = ["mobilenetv1", "mobilenetv2", "mobilenetv3", "shufflenetv1", "shufflenetv2", 'mobilenet_shuffle']
#result_files = ["./results/ssd_mobilenetv1_result.txt", "./results/ssd_mobilenetv2_result.txt", "./results/ssd_mobilenetv3_result.txt",
#               "./results/ssd_shufflenetv1_group3_result.txt", "./results/ssd_shufflenetv2_result.txt", "./results/ssd_shuffle_mobilenet_result.txt"]

weight_paths = ["./ssd_vgg/ssd_adam_vgg-147.h5"]
networks = ["vgg"]
result_files = ["./ssdlite_adam_vgg_result.txt"]
#print(len(predicts))
x = np.zeros((31), dtype=np.int8)
for index, weight_path in enumerate(weight_paths):
    det = detector(weight_path=weight_path, network=networks[index], groups=3)
    result_file = open(result_files[index], "w")

    for index, pre in enumerate(predicts):
        #start_time = datetime.datetime.now()
        img = pre[0]
        print(img)
        
        image = Image.open(img)
        #image = det.detect_image(image)
        #image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        #end_time = datetime.datetime.now()
        #print("{}".format(end_time-start_time))
        #cv2.imshow('TEST', image)
        #cv2.waitKey(0)
        
        results = det.detected_bbox(image)
        #for result in results:
        #    if result[4] == 20:
        #        x[int(result[4])] += 1
        
        result_file.write(img)
        for result in results:
            result_file.write(" {},{},{},{},{}".format(result[0], result[1], result[2], result[3], result[4]))
        result_file.write("\n")
       
        #result_file.write("\n")
        
    #result_file.close()
