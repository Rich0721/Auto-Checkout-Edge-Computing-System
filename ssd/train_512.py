from __future__ import annotations
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from ssd_keras_layers.ModelCheckpoint import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from models.ssd_512 import SSD512
from loss.ssd_loss import ssd_loss
from generator import Generator
from utils.anchors import get_anchors_512
from utils.utils import BBoxUtility
from config import config


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if not os.path.exists(config.MODEL_FOLDER):
    os.mkdir(config.MODEL_FOLDER)

if __name__ == "__main__":

    priors = get_anchors_512((config.IMAGE_SIZE_512[0], config.IMAGE_SIZE_512[1]))
    
    bbox_util = BBoxUtility(len(config.CLASSES), priors)

    model = SSD512(config.IMAGE_SIZE_512,
                    n_classes=len(config.CLASSES),
                    anchors=config.ANCHORS_SIZE_512,
                    variances=config.VARIANCES)

    checkpoint = ModelCheckpoint(config.MODEL_FOLDER + config.FILE_NAME + "{epoch:02d}.h5",
                monitor='val_loss', save_weights_only=True, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    
    val_split = 0.1
    with open(config.TRAIN_TEXT) as f:
        lines = f.readlines()
    np.random.seed(1000)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    gen = Generator(bbox_util, config.BATCH_SIZE, lines[:num_train], lines[num_train:], (config.IMAGE_SIZE_512[0], config.IMAGE_SIZE_512[1]), len(config.CLASSES))

    model.compile(optimizer=Adam(lr=5e-4), loss=ssd_loss(len(config.CLASSES)).compute_loss)
    model.fit_generator(gen.generator(True),
                        steps_per_epoch=num_train//config.BATCH_SIZE,
                        validation_data=gen.generator(False),
                        validation_steps=num_val//config.BATCH_SIZE,
                        epochs=config.EPOCHS,
                        initial_epoch=0,
                        callbacks=[checkpoint, reduce_lr])
    model.save(config.MODEL_FOLDER, save_format='tf')