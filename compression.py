import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import input_data
import time

SUMMARY_DIR = '/tensorflow/cifar/compression19'

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
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

def show_validation_images(orig, pred, title, n = 6, show_orig=True):

    plt.figure(figsize=(6, 2))
    if show_orig:
        f, axs = plt.subplots(2, 6)
    else:
        f, axs = plt.subplots(1, 6)

    for i in range(n):
        if show_orig:
            ax_orig = axs[1, i]
            ax_orig.imshow(orig[i])
            ax_orig.get_xaxis().set_visible(False)
            ax_orig.get_yaxis().set_visible(False)

            ax_pred = axs[0, i]
            ax_pred.imshow(pred[i])
            ax_pred.get_xaxis().set_visible(False)
            ax_pred.get_yaxis().set_visible(False)
        else:
            ax_pred = axs[i]
            ax_pred.imshow(pred[i])
            ax_pred.get_xaxis().set_visible(False)
            ax_pred.get_yaxis().set_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(title)
    plt.show()
    return

def encoder(x, kernel_size, num_filters, stride, output_size):
    conv_w = int((IMAGE_WIDTH - kernel_size) / stride) + 1
    conv_h = int((IMAGE_HEIGHT - kernel_size) / stride) + 1

    with tf.name_scope('encoder'):
        # Single layer convolution
        conv_1 = tf.layers.conv2d(x, num_filters, [kernel_size, kernel_size], [stride, stride], activation=tf.nn.sigmoid, name='conv_1', kernel_initializer=tf.truncated_normal_initializer)
        flat = tf.reshape(conv_1, [-1, num_filters * conv_w * conv_h], name='flatten')
        layer_1 = tf.layers.dense(flat, output_size, activation=tf.nn.sigmoid, name='fc_layer')

        # Single dense layer
        #flat = tf.reshape(x, [-1, input_size], name='flatten')
        #layer_1 = tf.layers.dense(flat, hidden_layer, activation=tf.nn.sigmoid, name='fc_layer')
        return layer_1


def decoder(x, kernel_size, num_filters, stride):
    conv_w = int((IMAGE_WIDTH - kernel_size) / stride) + 1
    conv_h = int((IMAGE_HEIGHT - kernel_size) / stride) + 1

    with tf.name_scope('decoder'):
        # Single layer convolution
        layer_1 = tf.layers.dense(x, num_filters * int(conv_w) * int(conv_h), activation=tf.nn.sigmoid, name='conv_trans_input')
        conv_input = tf.reshape(layer_1, [-1, int(conv_h), int(conv_w), num_filters], name='reshape')
        conv_1 = tf.layers.conv2d_transpose(conv_input, 3, [kernel_size, kernel_size], [stride, stride], activation=tf.nn.sigmoid, name='conv_trans', kernel_initializer=tf.truncated_normal_initializer)

        # Single dense layer
        #layer_1 = tf.layers.dense(x, input_size, activation=tf.nn.sigmoid, name='fc_layer_2')
        #reshape = tf.reshape(layer_1, [-1, width, height, channels])

    return conv_1


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


def compression_model(num_filters, loss_fc, lr, cr):
    hparam_str = '/f=' + str(num_filters) + ',loss=' + loss_fc + ',lr=' + str(lr) + ',cr=' + str(cr)
    print('RUN: ' + hparam_str)

    tf.reset_default_graph()

    output_size = int(INPUT_SIZE / cr)

    kernel_size = 8
    stride = 1

    # Setup placeholders
    X = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS], name='X')

    # Setup encoder and decoder layers
    encoder_op = encoder(X, kernel_size, num_filters, stride, output_size)
    decoder_op = decoder(encoder_op, kernel_size, num_filters, stride)

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
            train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(mse)
        elif loss_fc == 'psnr':
            train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(-psnr)
        else:
            train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(ssim_loss)

    merged = tf.summary.merge_all()

    size = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print(size)

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(SUMMARY_DIR + hparam_str + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(SUMMARY_DIR + hparam_str + '/test', sess.graph)

        sess.run(tf.global_variables_initializer())

        data_train_a = input_data.get_cifar_train(16)
        iterator_train_a = data_train_a.make_one_shot_iterator()
        next_element_train_a = iterator_train_a.get_next()

        data_test = input_data.get_cifar_test(64)
        iterator_test = data_test.make_one_shot_iterator()
        next_element_test = iterator_test.get_next()

        data_valid = input_data.get_cifar_validation(6)
        iterator_valid = data_valid.make_one_shot_iterator()
        next_element_valid = iterator_valid.get_next()

        validation_images = sess.run(next_element_valid);

        # Amount of time to train model in seconds
        timeout = time.time() + 60*60

        step = 0

        while True:
            train_batch = sess.run(next_element_train_a)
            summary, _ = sess.run([merged, train_step], feed_dict={X: train_batch})

            if step % 25 == 0:
                train_writer.add_summary(summary, step)

            if step % 100 == 0:
                test_b = sess.run(next_element_test)

                orig = test_b
                summary, pred, current_loss = sess.run([merged, decoder_op, mse], feed_dict={X: test_b})
                test_writer.add_summary(summary, step)

                print('Step ' + str(step) + ', current loss: ' + str(current_loss))

            if step % 50000 == 0:
                summary, pred = sess.run([merged, decoder_op], feed_dict={X: validation_images})
                show_validation_images(validation_images[0:6], pred[0:6], 'Step ' + str(step))

            # Increment step
            step += 1

            if time.time() > timeout:
                summary, pred, m, p, s = sess.run([merged, decoder_op, mse, psnr, ssim_loss], feed_dict={X: validation_images})
                show_validation_images(validation_images[0:6], pred[0:6], 'Step ' + str(step) + ' - ' + hparam_str + '\npsnr = ' + str(p) + 'dB, mse = ' + str(m) + ', ssim = ' + str(1 - s), show_orig=False)
                break


def main():

    # Try different loss functions
    for cr in [16, 32, 96]:
            for loss_fc in ['psnr', 'ssim', 'mse']:
                compression_model(32, loss_fc, 0.0001, cr)


if __name__ == '__main__':
    main()
