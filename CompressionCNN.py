import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import input_data
import time

SUMMARY_DIR = '/tensorflow/compression104'

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS

def show_images(left, right, title):
    """ Displays two side images side by side.

    :param left: The left image
    :param right: The right image
    :param title: The title
    :return: None
    """
    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(left)
    ax1.set_title("Original")

    ax2.imshow(right)
    ax2.set_title("Predicted")

    plt.suptitle(title)
    plt.show()
    return

def encoder(x, kernel_size, num_filters, stride, output_size):
    conv_w = int((IMAGE_WIDTH - kernel_size) / stride) + 1
    conv_h = int((IMAGE_HEIGHT - kernel_size) / stride) + 1

    with tf.name_scope('encoder'):
        convolution_layer_a = tf.layers.conv2d(x, num_filters, [kernel_size, kernel_size], [stride, stride],
                                               activation=tf.nn.sigmoid, name='convolution_layer_a',
                                               kernel_initializer=tf.truncated_normal_initializer)
        average_pooling_layer_a = tf.layers.average_pooling2d(convolution_layer_a, [2, 2], [2, 2],
                                                            name='average_pooling_layer_a')
        convolution_layer_b = tf.layers.conv2d(average_pooling_layer_a, 3, [3, 3], [1, 1],
                                               activation=tf.nn.sigmoid, name='convolution_layer_b',
                                               kernel_initializer=tf.truncated_normal_initializer)
        average_pooling_layer_b = tf.layers.average_pooling2d(convolution_layer_b, [2, 2], [2, 2],
                                                            name='average_pooling_layer_b')
        # flatten_layer = tf.reshape(average_pooling_layer_b, [-1, num_filters * conv_w * conv_h], name='flatten_layer')
        # fc_layer = tf.layers.dense(flatten_layer, output_size, activation=tf.nn.sigmoid, name='fc_layer_encoder')
        return average_pooling_layer_b


def decoder(x, kernel_size, num_filters, stride):
    conv_w = int((IMAGE_WIDTH - kernel_size) / stride) + 1
    conv_h = int((IMAGE_HEIGHT - kernel_size) / stride) + 1

    with tf.name_scope('decoder'):
        deconvolution_layer_a = tf.layers.conv2d_transpose(x, 48, [7, 7], [2, 2], activation=tf.nn.sigmoid,
                                                         name='deconvolution_layer_a',
                                                         kernel_initializer=tf.truncated_normal_initializer)
        deconvolution_layer_b = tf.layers.conv2d_transpose(deconvolution_layer_a, 3, [8, 8], [2, 2], activation=tf.nn.sigmoid,
                                                           name='deconvolution_layer_b',
                                                           kernel_initializer=tf.truncated_normal_initializer)
        # Single layer convolution
        #fc_layer = tf.layers.dense(x, num_filters * int(conv_w) * int(conv_h), activation=tf.nn.sigmoid,
        #                           name='fc_layer_decoder')
        #deconvolution_input_layer = tf.reshape(fc_layer, [-1, int(conv_w), int(conv_h), num_filters],
        #                                       name='deconvolution_input_layer')
        #deconvolution_layer = tf.layers.conv2d_transpose(deconvolution_input_layer, 3, [kernel_size, kernel_size],
         #                                                [stride, stride], activation=tf.nn.sigmoid,
         #                                                name='deconvolution_layer',
         #                                                kernel_initializer=tf.truncated_normal_initializer)
    return deconvolution_layer_b


def ssim(orig_b, pred_b):
    """Calculate the ssim difference between two batches of images. gives value 0f to 2f, 0 is an exact match.

    :param orig_b: original images
    :param pred_b: predicted images
    :return: the average ssim difference between both batches
    """

    def make_average_filter(size):
        """Makes an average filter of size: size x size x channels

        :param size: the size of the filter
        :return: and averaging filter
        """
        avg_filter = np.ones([size, size, IMAGE_CHANNELS, 1], dtype=np.float32)
        avg_filter = avg_filter / (size * size * IMAGE_CHANNELS)
        avg_filter = tf.constant(avg_filter)
        return avg_filter

    # These are the default values for calculating SSIM
    k1 = 0.01
    k2 = 0.03
    n = 8
    c1 = k1 ** 2
    c2 = k2 ** 2

    average = make_average_filter(n)

    orig_avg = tf.nn.conv2d(orig_b, average, [1, 8, 8, 1], padding='VALID')
    pred_avg = tf.nn.conv2d(pred_b, average, [1, 8, 8, 1], padding='VALID')

    orig_avg_sq = orig_avg ** 2
    pred_avg_sq = pred_avg ** 2

    orig_sq = tf.square(orig_b)
    pred_sq = tf.square(pred_b)
    orig_pred = tf.multiply(orig_b, pred_b)

    orig_var = tf.nn.conv2d(orig_sq, average, [1, 8, 8, 1], padding='VALID') - (orig_avg_sq / (n ** 2))
    pred_var = tf.nn.conv2d(pred_sq, average, [1, 8, 8, 1], padding='VALID') - (pred_avg_sq / (n ** 2))
    co_var = tf.nn.conv2d(orig_pred, average, [1, 8, 8, 1], padding='VALID') - ((orig_avg * pred_avg) / (n ** 2))

    orig_var = orig_var / ((n ** 2) - 1)
    pred_var = pred_var / ((n ** 2) - 1)
    co_var = co_var / ((n ** 2) - 1)

    value = ((2 * orig_avg * pred_avg + c1) * (2 * co_var + c2)) / \
            ((orig_avg_sq + pred_avg_sq + c1) * (orig_var + pred_var + c2))

    ans = tf.reduce_mean(value)

    # SSIM actually gives a value from -1.0f to 1.0f, but make 0 to 2.0f is easier to understand.
    return 1 - ans


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def compression_model(num_filters, loss_fc, max_a_steps, max_b_steps):
    hparam_str = '/f=' + str(num_filters) + ',loss=' + loss_fc + ',a-steps=' + str(max_a_steps) + ',b-steps=' + str(max_b_steps)
    print('RUN: ' + hparam_str)

    tf.reset_default_graph()

    output_size = int(INPUT_SIZE / 48)

    kernel_size = 7
    stride = 3

    # Setup placeholders
    X = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS], name='X')

    # Setup encoder and decoder layers
    encoder_op = encoder(X, 11, num_filters, 1, output_size)
    decoder_op = decoder(encoder_op, 7, num_filters, 1)

    y_pred = decoder_op
    y_true = X

    with tf.name_scope('accuracy'):
        with tf.name_scope('ssim'):
            # Calculate the SSIM of the predicted and original image.
            ssim_loss = ssim(y_true, y_pred)
        with tf.name_scope('mse'):
            # Calculate the MSE of the predicted and original image.
            mse = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        with tf.name_scope('psnr'):
            psnr = 10 * log10(1 / mse)

    # Add these values to log.
    tf.summary.scalar('ssim', ssim_loss)
    tf.summary.scalar('mse', mse)
    tf.summary.scalar('psnr', psnr)

    with tf.name_scope('train'):
        if loss_fc == 'mse':
            train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(mse)
        elif loss_fc == 'psnr':
            train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(-psnr)
        else:
            train_step = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(ssim_loss)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(SUMMARY_DIR + hparam_str + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(SUMMARY_DIR + hparam_str + '/test', sess.graph)

        test_batch_size = 64

        MAX_A_TRAIN_SIZE = 500
        MAX_B_TRAIN_SIZE = 500
        MAX_C_TRAIN_SIZE = 50000

        MAX_VALID_SIZE = 1000

        size = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print(size);

        sess.run(tf.global_variables_initializer())

        data_train_a = input_data.get_data(8, MAX_A_TRAIN_SIZE, True)
        iterator_train_a = data_train_a.make_one_shot_iterator()
        next_element_train_a = iterator_train_a.get_next()

        data_train_b = input_data.get_data(16, MAX_B_TRAIN_SIZE, True)
        iterator_train_b = data_train_b.make_one_shot_iterator()
        next_element_train_b = iterator_train_b.get_next()

        data_train_c = input_data.get_data(128, MAX_C_TRAIN_SIZE, True)
        iterator_train_c = data_train_c.make_one_shot_iterator()
        next_element_train_c = iterator_train_c.get_next()

        data_valid = input_data.get_data(test_batch_size, MAX_VALID_SIZE, False)
        iterator_valid = data_valid.make_one_shot_iterator()
        next_element_valid = iterator_valid.get_next()

        # Amount of time to train model in seconds
        timeout = time.time() + 60*90

        step = 0

        while True:
            if step <= max_a_steps:
                train_batch = sess.run(next_element_train_a)
                summary, _ = sess.run([merged, train_step], feed_dict={X: train_batch})
            elif step <= max_b_steps:
                train_batch = sess.run(next_element_train_b)
                summary, _ = sess.run([merged, train_step], feed_dict={X: train_batch})
            else:
                train_batch = sess.run(next_element_train_c)
                summary, _ = sess.run([merged, train_step], feed_dict={X: train_batch})

            if step % 25 == 0:
                train_writer.add_summary(summary, step)

            if step % 100 == 0:
                test_b = sess.run(next_element_valid)

                orig = test_b
                summary, pred, coded, current_loss = sess.run([merged, decoder_op, encoder_op, mse], feed_dict={X: test_b})
                test_writer.add_summary(summary, step)

                print('Step ' + str(step) + ', current loss: ' + str(current_loss))

            if step % 500 == 0:
                show_images(orig[0], pred[0], hparam_str + ' - Step ' + str(step))
                show_images(orig[0], coded[0], 'coded')

            # Increment step
            step += 1

            if time.time() > timeout:
                show_images(orig[0], pred[0], hparam_str + ' - Step ' + str(step))
                break

def main():

    # Try different loss functions
    for filters in [48, 16]:
        for loss_fc in ['ssim', 'psnr']:
            compression_model(filters, loss_fc, 0, 0)


if __name__ == '__main__':
    main()