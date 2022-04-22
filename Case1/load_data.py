from pathlib import Path
# %tensorflow_version 1.x
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data():
    _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)

    base_dir = Path(zip_dir).parent / "cats_and_dogs_filtered"
    train_dir = base_dir / 'train'
    validation_dir = base_dir / 'validation'

    return train_dir, validation_dir


def run():
    res = load_data()

    train_cats_dir = res[0] / 'cats'
    train_dogs_dir = res[0] / 'dogs'
    validation_cats_dir = res[1] / 'cats'
    validation_dogs_dir = res[1] / 'dogs'

    num_cats_tr = len(list(train_cats_dir.glob("*")))
    num_dogs_tr = len(list(train_dogs_dir.glob("*")))

    num_cats_val = len(list(validation_cats_dir.glob("*")))
    num_dogs_val = len(list(validation_dogs_dir.glob("*")))

    total_train = num_cats_tr + num_dogs_tr
    total_val = num_cats_val + num_dogs_val

    print("--")
    print('total training cat images:', num_cats_tr)
    print('total training dog images:', num_dogs_tr)
    print("--")
    print('total validation cat images:', num_cats_val)
    print('total validation dog images:', num_dogs_val)
    print("--")
    print("Total training images:", total_train)
    print("Total validation images:", total_val)
    print("--")


if __name__ == "__main__":

    run()
