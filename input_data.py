import glob
import tensorflow as tf

TRAIN_PATH = 'var/data/train_64x64/'
VALID_PATH = 'var/data/valid_64x64/'
FILE_SUFFIX = '.png'

MAX_TRAIN_SIZE = 10000
MAX_VALID_SIZE = 1000

NUM_CORES = 8


def decode_image(image_path):
    image = tf.read_file(image_path)
    image_decoded = tf.image.decode_png(image)
    image_converted = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    return image_converted


def get_filenames(train=True):
    filenames = []

    if train:
        for image_path in glob.glob(TRAIN_PATH + '*' + FILE_SUFFIX):
            filenames.append(tf.constant(image_path))
            if len(filenames) >= MAX_TRAIN_SIZE:
                return filenames
    else:
        for image_path in glob.glob(VALID_PATH + '*' + FILE_SUFFIX):
            filenames.append(tf.constant(image_path))
            if len(filenames) > MAX_VALID_SIZE:
                return filenames

    return


def get_data(batch_size, train=True):
    dataset = tf.data.Dataset.from_tensor_slices(get_filenames(train))
    # It goes faster if we don't shuffle the dataset.
    dataset = dataset.shuffle(batch_size * 10)
    dataset = tf.data.Dataset.map(dataset, lambda path: decode_image(path), NUM_CORES)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(1)
    return dataset

