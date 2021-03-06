
class config:

    IMAGE_SIZE = (416, 416)
    ANCHORS = [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326] # Original VOC datasets
    '''
    CLASSES = ['1402200300101', '1402300300101', '1402310200101', '1402312700101', '1402312900101', 
                '1402324800101', '1422001900101', '1422111300101', '1422204600101', '1422206800101', 
                '1422300200101', '1422300300101', '1422301800101', '1422302000101', '1422305100101', 
                '1422305900101', '1422308000101', '1422329600101', '1422503600101', '1422504400101', 
                '1422505200101', '1422505600101', '1422593400101', '1422594600101', '1423003100101', 
                '1423014700101', '1423100700101', '1423103600101', '1423206800101', '1423207800101', '1423301600101']
    '''
    CLASSES = ['person', 'bicycle','car','motorcycle','airplane','bus','train','truck', 'boat',
            'traffic light', 'fire hydrant','stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
            'baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
            'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote',
            'keyboard','cell phone', 'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
    #text_file = open("insert_classes.txt").readlines()

    #CLASSES = [c[:-1] for c in text_file]
    ##############Train####################
    TRAIN_TEXT = ["./train.txt", './val.txt',"./test.txt"]
    DATASET = "../datasets/test_network/"
    VOC_TEXT_FILE = ["../datasets/test_network/train.txt","../datasets/test_network/val.txt", "../datasets/test_network/test.txt"]
    COCO_JSON = ['instances_train2014.json', 'instances_val2014.json']
    COCO_TRAIN_TEXT = ["./train_coco.txt", "./test_coco.txt"]
    COCO_DATASER_FOLDER = ["../datasets/coco_train2014/", "../datasets/COCO_val2014/"]
    
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-3
    EPOCHS = 150
    
    IGNORE_THRESH = 0.5

    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    TENSORBOARD_DIR = "./yolov3_sgd"
    WEIGHTS_FILE = "yolov3"