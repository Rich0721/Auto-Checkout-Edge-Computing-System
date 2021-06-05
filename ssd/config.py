
class config:

    EPOCHS = 150
    BATCH_SIZE = 8
    IMAGE_SIZE_300 = (300, 300, 3)
    IMAGE_SIZE_512 = (512, 512, 3)
    ANCHORS_SIZE_300 = [30, 60, 111, 162, 213, 264, 315] # VOC SSD300
    #ANCHORS_SIZE_300 = [21, 45, 99, 153, 207, 261, 315] # COCO
    ANCHORS_SIZE_512 = [36, 77, 154, 230, 307, 384, 461, 538] # VOC SSD512
    #ANCHORS_SIZE_512 = [20, 51, 133, 215, 297, 379, 461, 542] # COCO
    VARIANCES = [0.1, 0.1, 0.2, 0.2]

    CLASSES = ['1402200300101', '1402300300101', '1402310200101', '1402312700101', '1402312900101', 
                '1402324800101', '1422001900101', '1422111300101', '1422204600101', '1422206800101',
                '1422300200101', '1422300300101', '1422301800101', '1422302000101', '1422305100101', 
                '1422305900101', '1422308000101', '1422329600101', '1422503600101', '1422504400101', 
                '1422505200101', '1422505600101', '1422593400101', '1422594600101', '1423003100101', 
                '1423014700101', '1423100700101', '1423103600101', '1423206800101', '1423207800101', '1423301600101']
                
    '''
    DATASET = ["../datasets/VOC2007/", "../datasets/VOC2012/"]
    VOC_TEXT_FILE = ["../datasets/VOC2007/trainval.txt", "../datasets/test_network/val.txt", "../datasets/test_network/test.txt"]
    VOC_TRAIN_FILE = ["./train.txt", "./val.txt", "./test.txt"]
    '''

    DATASET = ["../datasets/test_network/"]
    VOC_TEXT_FILE = ["../datasets/test_network/train.txt", "../datasets/test_network/val.txt", "../datasets/test_network/test.txt"]
    VOC_TRAIN_FILE = ["./train.txt", "./val.txt", "./test.txt"]

    COCO_JSON = ['instances_train2014.json', 'instances_val2014.json']
    COCO_DATASER_FOLDER = "../datasets/coco_train2014/"
    COCO_TRAIN_TEXT = ["./train.txt", "./val.txt"]

    CONFIDENCE = 0.5
    NMS_IOU = 0.4
    # "./ssd_mobilenetv1_voc2007/"
    #MODEL_FOLDER = "./ssd_vgg/"
    #FILE_NAME = "ssd_vgg"
    
    MODEL_FOLDER = ["./ssdlite_mobilenetv1/", "./ssdlite_mobilenetv2/", "./ssdlite_shuffle_mobilenet/", "./ssdlite_shufflenetv1/", "./ssdlite_shufflenetv2/"]
    FILE_NAME = ["ssdlite_adam_mobilenetv1", "ssdlite_adam_mobilenetv2", "ssdlite_adam_shuffle_mobilenet", "ssdlite__adamshufflenetv1", "ssdlite_adam_shufflenetv2"]
    