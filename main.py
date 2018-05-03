import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import input_data

SUMMARY_DIR = '/tensorflow/compression21'

width = 64
height = 64
channels = 3

input_size = width * height * channels
hidden_layer = int(input_size / 24)

kernel_size = 8
filters = 16
stride = 2

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


def encoder(x):
    conv_w = int((width - kernel_size) / stride) + 1
    conv_h = int((height - kernel_size) / stride) + 1

    with tf.name_scope('encoder'):
        # Single layer convolution
        conv_1 = tf.layers.conv2d(x, filters, [kernel_size, kernel_size], [stride, stride], activation=tf.nn.sigmoid, name='conv_1')
        flat = tf.reshape(conv_1, [-1, filters * conv_w * conv_h], name='flatten')
        layer_1 = tf.layers.dense(flat, hidden_layer, activation=tf.nn.sigmoid, name='fc_layer')

        # Single dense layer
        #flat = tf.reshape(x, [-1, input_size], name='flatten')
        #layer_1 = tf.layers.dense(flat, hidden_layer, activation=tf.nn.sigmoid, name='fc_layer')
        return layer_1


def decoder(x):
    conv_w = int((width - kernel_size) / stride) + 1
    conv_h = int((height - kernel_size) / stride) + 1

    with tf.name_scope('decoder'):
        # Single layer convolution
        layer_1 = tf.layers.dense(x, filters * int(conv_w) * int(conv_h), activation=tf.nn.sigmoid, name='conv_trans_input')
        conv_input = tf.reshape(layer_1, [-1, int(conv_w), int(conv_h), filters], name='reshape')
        conv_1 = tf.layers.conv2d_transpose(conv_input, 3, [kernel_size, kernel_size], [stride, stride], activation=tf.nn.sigmoid, name='conv_trans')

        # Single dense layer
        #layer_1 = tf.layers.dense(x, input_size, activation=tf.nn.sigmoid, name='fc_layer_2')
        #reshape = tf.reshape(layer_1, [-1, width, height, channels])

    return conv_1


def conv_layer(input, kernel, name='conv_layer'):
    with tf.name_scope(name):
        w = kernel
        b = tf.Variable(tf.constant(0.1, shape=[filters]))
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='VALID')
        act = tf.nn.sigmoid(conv + b)

        tf.summary.histogram("Weights", w)

        unstacked = tf.unstack(w, axis=3)
        stacked = tf.stack(unstacked, axis=0)

        tf.summary.image('Filters', stacked, max_outputs=filters)
        tf.summary.histogram("Bias", b)
        tf.summary.histogram('Activations', act)
    return act


def de_conv_layer(input, kernel, name='de_conv_layer'):
    with tf.name_scope(name):
        with tf.name_scope('Shape'):
            input_shape = tf.shape(input)
            batch_size = input_shape[0]

        w = kernel
        b = tf.Variable(tf.constant(0.1, shape=[channels]))
        de_conv = tf.nn.conv2d_transpose(input, w, [batch_size, 64, 64, channels], [1, 1, 1, 1], padding='VALID')
        act = tf.nn.sigmoid(de_conv + b)

        tf.summary.histogram('Bias', b)
        tf.summary.histogram('Activations', act)
    return act


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
        avg_filter = np.ones([size, size, channels, 1], dtype=np.float32)
        avg_filter = avg_filter / (size * size * channels)
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


# Setup placeholders
X = tf.placeholder(tf.float32, shape=[None, width, height, channels], name='X')

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X


with tf.name_scope('accuracy'):
    with tf.name_scope('ssim'):
        # Calculate the SSIM of the predicted and original image.
        ssim_loss = ssim(y_true, y_pred)
    with tf.name_scope('mse'):
        # Calculate the MSE of the predicted and original image.
        mse = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    with tf.name_scope('accuracy'):
        # The loss function used in training the network.
        loss = mse

    # Add these values to log.
    tf.summary.scalar('ssim', ssim_loss)
    tf.summary.scalar('mse', mse)
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer().minimize(loss)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_batch_size = 128
    test_batch_size = 256
    max_steps = 1000000

    train_writer = tf.summary.FileWriter(SUMMARY_DIR + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(SUMMARY_DIR + '/test', sess.graph)

    sess.run(tf.global_variables_initializer())

    data_train = input_data.get_data(train_batch_size, True)
    iterator_train = data_train.make_one_shot_iterator()
    next_element_train = iterator_train.get_next()

    data_valid = input_data.get_data(test_batch_size, False)
    iterator_valid = data_valid.make_one_shot_iterator()
    next_element_valid = iterator_valid.get_next()

    # Get the intial values for the kernel from

    for i in range(0, max_steps + 1):
        train_b = sess.run(next_element_train)

        summary, _ = sess.run([merged, train_step], feed_dict={X: train_b})
        train_writer.add_summary(summary, i)

        if i % 50 == 0:
            test_b = sess.run(next_element_valid)

            orig = test_b
            summary, pred, current_loss = sess.run([merged, decoder_op, loss], feed_dict={X: test_b})
            test_writer.add_summary(summary, i)

            print('Step ' + str(i) + ', current loss: ' + str(current_loss))

            if i % 1000 == 0:
                show_images(orig[0], pred[0], 'Step ' + str(i))



