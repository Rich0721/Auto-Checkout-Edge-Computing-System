import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import SGD, Adam
from loss.ssd_loss import ssd_loss

from generator import Generator
#from utils.anchors import get_anchors_300
from utils.anchors_mobilenet import get_anchors_300
from utils.utils import BBoxUtility
from config import config
import gc

def learning_rate(epoch):
    if epoch < 100:
        return 1e-3
    else:
        return 1e-4


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == "__main__":

    for index, folder in enumerate(config.MODEL_FOLDER):

        if index == 0:
            from models.ssd_mobilenetv1_300 import SSD300
        '''
        elif index == 1:
            from models.ssd_mobilenetv2_300 import SSD300
        elif index == 2:
            from models.ssd_mobilenetv2_split_300 import SSD300
        elif index == 3:
            from models.ssd_shufflenetv1_300 import SSD300
        elif index == 4:
            from models.ssd_shufflenetv2_300 import SSD300
        if not os.path.exists(folder):
            os.mkdir(folder)
        '''

        priors = get_anchors_300((config.IMAGE_SIZE_300[0], config.IMAGE_SIZE_300[1]))

        bbox_util = BBoxUtility(len(config.CLASSES), priors)

        model = SSD300(config.IMAGE_SIZE_300,
                        n_classes=len(config.CLASSES),
                        anchors=config.ANCHORS_SIZE_300,
                        variances=config.VARIANCES)
        model.summary()
        checkpoint = ModelCheckpoint(folder + config.FILE_NAME[index] + "-{epoch:02d}.h5",
                    monitor='val_loss', save_weights_only=True, save_best_only=True)
        reduce_lr = LearningRateScheduler(learning_rate, verbose=1)

        with open(config.VOC_TRAIN_FILE[0]) as f:
            lines = f.readlines()

        with open(config.VOC_TRAIN_FILE[1]) as f:
            val_lines = f.readlines()
        np.random.seed(1000)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_train = len(lines)
        num_val = len(val_lines)

        gen = Generator(bbox_util, config.BATCH_SIZE, lines, val_lines, (config.IMAGE_SIZE_300[0], config.IMAGE_SIZE_300[1]), len(config.CLASSES))

        model.compile(optimizer=Adam(lr=1e-3), loss=ssd_loss(len(config.CLASSES)).compute_loss)
        model.fit_generator(gen.generator(True),
                            steps_per_epoch=num_train//config.BATCH_SIZE,
                            validation_data=gen.generator(False),
                            validation_steps=num_val//config.BATCH_SIZE,
                            epochs=config.EPOCHS,
                            callbacks=[checkpoint, reduce_lr])
        model.save(folder, save_format='tf')
        gc.collect()