import glob
import tensorflow as tf
import numpy as np

TRAIN_PATH = 'var/data/oid_test_256_192/'
VALID_PATH = 'var/data/oid_validation_256_192/'
FILE_SUFFIX = '.jpg'

NUM_CORES = 8


def decode_image(image_path):
    image = tf.read_file(image_path)
    image_decoded = tf.image.decode_jpeg(image)
    image_converted = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    return image_converted


def get_filenames(data_size, train=True):
    filenames = []

    if train:
        for image_path in glob.glob(TRAIN_PATH + '*' + FILE_SUFFIX):
            filenames.append(tf.constant(image_path))
            if len(filenames) >= data_size:
                return filenames
    else:
        for image_path in glob.glob(VALID_PATH + '*' + FILE_SUFFIX):
            filenames.append(tf.constant(image_path))
            if len(filenames) > data_size:
                return filenames

    return


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_cifar_train(batch_size):
    data_batches = ['var/data/cifar-10-batches-py/data_batch_1',
                    'var/data/cifar-10-batches-py/data_batch_2']

    all_images = np.zeros(shape=(0, 32, 32, 3))

    for file in data_batches:
        datadict = unpickle(file)
        images_np = datadict[b'data']
        images = convert_cifar_images(images_np)
        all_images = np.append(all_images, images, axis=0)

    dataset = tf.data.Dataset.from_tensor_slices(all_images)
    dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(1)

    return dataset


def get_cifar_test(batch_size):
    data_batches = ['var/data/cifar-10-batches-py/test_batch']

    all_images = np.zeros(shape=(0, 32, 32, 3))

    for file in data_batches:
        datadict = unpickle(file)
        images_np = datadict[b'data']
        images = convert_cifar_images(images_np)
        all_images = np.append(all_images, images, axis=0)

    dataset = tf.data.Dataset.from_tensor_slices(all_images)
    dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(1)

    return dataset


def get_cifar_validation(n):
    data_batches = ['var/data/cifar-10-batches-py/test_batch']

    all_images = np.zeros(shape=(0, 32, 32, 3))

    for file in data_batches:
        datadict = unpickle(file)
        images_np = datadict[b'data']
        images = convert_cifar_images(images_np)
        all_images = np.append(all_images, images, axis=0)

    dataset = tf.data.Dataset.from_tensor_slices(all_images)
    dataset = dataset.batch(n)
    dataset = dataset.prefetch(1)

    return dataset


def convert_cifar_images(raw):
    num_channels = 3
    img_size = 32

    raw_float = np.array(raw, dtype=float) / 255.0
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    images = images.transpose([0, 2, 3, 1])

    return images


def get_data(batch_size, data_size, train=True):
    dataset = tf.data.Dataset.from_tensor_slices(get_filenames(data_size, train))
    # It goes faster if we don't shuffle the dataset.
    dataset = dataset.shuffle(batch_size * 10)
    dataset = tf.data.Dataset.map(dataset, lambda path: decode_image(path), NUM_CORES)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(1)
    return dataset

