import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import load_data as ld


BATCH_SIZE = 100
IMG_SHAPE  = 150


def batches():
    train_dir, validation_dir = ld.load_data()

    train_image_generator      = ImageDataGenerator(rescale=1./255)
    validation_image_generator = ImageDataGenerator(rescale=1./255)

    train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=(IMG_SHAPE,IMG_SHAPE),
                                                               class_mode='binary')

    val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                                  directory=validation_dir,
                                                                  shuffle=False,
                                                                  target_size=(IMG_SHAPE,IMG_SHAPE),
                                                                  class_mode='binary')

    return train_data_gen, val_data_gen