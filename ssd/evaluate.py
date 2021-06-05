import numpy as np
from config import config
import os

IOU_THRESHOLD = 0.5

def compute_ap(recall, precision):
    
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    for i in range(mpre.size -1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
    
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])
    
    return ap
    
def compute_iou(b1, b2):

    inter_xmin = max(b1[0], b2[0])
    inter_ymin = max(b1[1], b2[1])
    inter_xmax = min(b1[2], b2[2])
    inter_ymax = min(b1[3], b2[3])
    
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)

    inter_area = inter_w * inter_h
    b1_area =  (b1[2] - b1[0]) * (b1[3] - b1[1])
    b2_area =  (b2[2] - b2[0]) * (b2[3] - b2[1])
    return max(0,inter_area / (b1_area + b2_area - inter_area))  


def compute_average_precision(init_file, result_file, iou_threshold=0.5):

    with open(init_file) as f:
        init_lines = f.readlines()

    with open(result_file) as f:
        result_lines = f.readlines()
    TP = np.zeros(len(config.CLASSES), dtype=np.int32)
    FP = np.zeros(len(config.CLASSES), dtype=np.int32)
    NUM_ANNOTATION = np.zeros(len(config.CLASSES), dtype=np.int32) 
    #print(all_detections)
    #f = open("test_error.txt", 'w')
    i = 0
    
    for index in range(len(result_lines)):

        original = init_lines[index].split()
        result = result_lines[index].split()
        original_index = []
        result_index = []

        original_bboxes = np.array([np.array(list(map(int, box.split(",")))) for box in original[1:]])
        result_bboxes = np.array([np.array(list(map(int, box.split(",")))) for box in result[1:]])
        
        
        for o, original_box in enumerate(original_bboxes):

            NUM_ANNOTATION[original_box[-1]]+=1
            for r, result_bbox in enumerate(result_bboxes):

                iou = compute_iou(original_box, result_bbox)
                if iou >= iou_threshold:           
                    result_index.append(r)
                    original_index.append(o)
                    if result_bbox[-1] == original_box[-1]:
                        TP[original_box[4]]+=1
                    else:
                        FP[original_box[4]]+=1
    
    #pre = TP / (TP+FP)
    recall = TP / NUM_ANNOTATION
    #print(NUM_ANNOTATION)
    
    for i in range(len(config.CLASSES)):
     
       print("{}:{:.3f}".format(config.CLASSES[i],recall[i]))
    
    #print(np.sum(pre)/31)
    print("{:.3f}".format(np.sum(recall)/31))
    #print("AP:{:.2f}".format(compute_ap(recall, pre)))
    

def union_commtity(init_file, result_file, iou_threshold=0.75):
    
    all = 0
    correct = 0
    with open(init_file) as f:
        init_lines = f.readlines()

    with open(result_file) as f:
        result_lines = f.readlines()
    
    for index in range(len(result_lines)):

        original = init_lines[index].split()
        result = result_lines[index].split()
        
        result_set = []
        original_set = []
        original_bboxes = np.array([np.array(list(map(int, box.split(",")))) for box in original[1:]])
        result_bboxes = np.array([np.array(list(map(int, box.split(",")))) for box in result[1:]])

        if original_bboxes.shape[0] == result_bboxes.shape[0]:
            for o, original_box in enumerate(original_bboxes):
                for r, result_bbox in enumerate(result_bboxes):
                    iou = compute_iou(original_box, result_bbox)
                    if iou >= iou_threshold:           
                        result_set.append(result_bbox[4])
                        original_set.append(original_box[4])
                        break
            result_set = set(result_set)
            original_set = set(result_set)
            intersection = original_set.intersection(result_set)
            union = original_set.union(result_set)
            if len(union) == len(intersection):
                correct+=1
        all+=1

    print("correct:{}, all:{}, av:{:.4f}".format(correct, all, correct/all))

if __name__ == "__main__":

    result_files = ['./ssdlite_adam_vgg_result.txt']
    #print(result_files)
    #result_files = ['ssdlite_mobilenetv1_result.txt', 'ssdlite_mobilenetv2_result.txt', 'ssdlite_mobilenetv3_result.txt', 
    #'ssdlite_shufflenetv1_group3_result.txt', 'ssdlite_shufflenetv2_result.txt', 'ssdlite_shuflle_mobilenet_result.txt', 
    #'ssd_mobilenetv1_result.txt', 'ssd_mobilenetv2_result.txt', 'ssd_mobilenetv3_result.txt', 
    #'ssd_shufflenetv1_group3_result.txt', 'ssd_shufflenetv2_result.txt', 'ssd_shuffle_mobilenet_result.txt']
    
    for index, result in enumerate(result_files):
        print(result)
        compute_average_precision(init_file="test.txt", result_file=result, iou_threshold=0.75)
        #union_commtity(init_file="test.txt", result_file="./results/" + result)
        print("========================")
    