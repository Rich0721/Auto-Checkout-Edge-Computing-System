import numpy as np
from pycocotools.coco import COCO
from config import config


def id2name(coco):
    classes = dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']
    return classes


def convert_annotation(image, coco, write_file, dir, class_dict):

    file_name = coco.loadImgs(image)[0]['file_name']
    print(file_name)
    #write_file.write(dir + file_name)
    
    anns_ids = coco.getAnnIds(image)
    anns = coco.loadAnns(anns_ids)
    
    if len(anns) > 0:
        write_file.write(dir + file_name)
    else:
        return False

    for ann in anns:
        class_name = class_dict[ann['category_id']]
        classes_index = config.CLASSES.index(class_name)
        if 'bbox' in ann:
            bbox = ann['bbox']
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[0] + bbox[2])
            ymax = int(bbox[1] + bbox[3])
            
            write_file.write(" {},{},{},{},{}".format(xmin, ymin, xmax, ymax, classes_index))
    return True

if __name__ == "__main__":
    
    write_train_file = open(config.COCO_TRAIN_TEXT[0], 'w')
    write_test_file = open(config.COCO_TRAIN_TEXT[1], 'w')

    for index, js in enumerate(config.COCO_JSON):

        coco = COCO(js)
        classes_dict = id2name(coco)
        
        coco_imgs = coco.imgs
        for i, image in enumerate(coco_imgs):
            if index == 0:
                line_up =  convert_annotation(image, coco, write_train_file, config.COCO_DATASER_FOLDER[index], classes_dict)
                if line_up:
                    write_train_file.write("\n")
            else:
                if np.random.randint(0, 2) == 0:
                    line_up =  convert_annotation(image, coco, write_train_file, config.COCO_DATASER_FOLDER[index], classes_dict)
                    if line_up:
                        write_train_file.write("\n")
                else:
                    line_up =  convert_annotation(image, coco, write_test_file, config.COCO_DATASER_FOLDER[index], classes_dict)
                    if line_up:
                        write_test_file.write("\n")
                

    write_train_file.close()
    write_test_file.close()