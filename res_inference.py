
import tensorflow as tf

DATA_FORMAT = 'channels_last'
NUM_CHANNELS = 3
GENDER_CLASS_NUM = 2
AGE_CLASS_NUM = 3

BLOCK1_DEEP = 32;
BLOCK2_DEEP = 64;
BLOCK3_DEEP = 128;
BLOCK4_DEEP = 256;
BLOCK5_DEEP = 512;

BLOCK_STRIDES = 2

BLOCK1_NUM = 1;
BLOCK2_NUM = 3;
BLOCK3_NUM = 4;
BLOCK4_NUM = 6;
BLOCK5_NUM = 3;

CONV1_DEEP = 32
CONV1_SIZE = 5

OUT1_DEEP = 512
OUT1_SIZE = 1

OUT2_DEEP = 512
OUT2_SIZE = 1

#OUT3_DEEP = 2
OUT3_SIZE = 1

FC_SIZE = 128

_BATCH_NORM_DECAY = 0.75
_BATCH_NORM_EPSILON = 0.0001
"""
def batch_norm(inputs, training):
    #Performs a batch normalization using a standard set of parameters.
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization(inputs=inputs, axis=3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)
"""
def batch_norm(inputs, training):
    #Performs a batch normalization using a standard set of parameters.
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    #with tf.variable_scope("Bnorm", reuse=tf.AUTO_REUSE):
    bnorm = tf.layers.batch_normalization(inputs=inputs, axis=3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)
    return bnorm
def fixed_conv2d(inputs, filters, kernel_size, strides):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME'), use_bias=True,kernel_initializer=tf.variance_scaling_initializer(),
        data_format=DATA_FORMAT)

def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,
    strides):
    """
    Similar to _building_block_v2(), except using the "bottleneck" blocks
    described in:
        Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

    adapted to the ordering conventions of:
        Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
    """
    shortcut = inputs
    inputs = batch_norm(inputs, training)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = fixed_conv2d(inputs=inputs, filters=filters, kernel_size=1, strides=1)
    inputs = batch_norm(inputs, training)
    inputs = tf.nn.relu(inputs)
    inputs = fixed_conv2d(inputs=inputs, filters=filters, kernel_size=3, strides=strides)

    inputs = batch_norm(inputs, training)
    inputs = tf.nn.relu(inputs)
    inputs = fixed_conv2d(inputs=inputs, filters=4 * filters, kernel_size=1, strides=1)

    return inputs + shortcut


def block_layer(inputs, filters, blocks_num, strides,training):
    """Creates one layer of blocks for the ResNet model.

    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on DATA_FORMAT.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    DATA_FORMAT: The input format ('channels_last' or 'channels_first').

    Returns:
    The output tensor of the block layer.
    """

    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4

    def projection_shortcut(inputs):
        return fixed_conv2d(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides)

  # Only the first block per block_layer uses projection_shortcut and strides
    inputs = _bottleneck_block_v2(inputs, filters, training, projection_shortcut, strides)

    for _ in range(1, blocks_num):
        inputs = _bottleneck_block_v2(inputs, filters, training, None, 1)

    return inputs



def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def inference(images,training):
    parameters = []
    #-------------------------------------------------------------------------
    # ResNet
    # block cnn layer 1
    with tf.variable_scope('block_cnn_1'):
        block1_temp = fixed_conv2d(inputs=images, filters=BLOCK1_DEEP, kernel_size=3, strides=1)
        batch_temp = batch_norm(block1_temp, training)
        relu_temp = tf.nn.relu(batch_temp)
        block1 = fixed_conv2d(inputs=relu_temp, filters=BLOCK1_DEEP, kernel_size=3, strides=2)
        print_activations(block1)
    # block cnn layer 2
    with tf.variable_scope('block_cnn_2'):
        block2 = block_layer(block1, BLOCK2_DEEP, BLOCK1_NUM, 2,training)
        print_activations(block2)
    # block cnn layer 3
    with tf.variable_scope('block_cnn_3'):
        block3 = block_layer(block2, BLOCK3_DEEP, BLOCK2_NUM, 2,training)
        print_activations(block3)
    # block cnn layer 4
    with tf.variable_scope('block_cnn_4'):
        block4 = block_layer(block3, BLOCK4_DEEP, BLOCK3_NUM, 2,training)
        print_activations(block4)
    # block cnn layer 5
    with tf.variable_scope('block_cnn_5'):
        block5 = block_layer(block4, BLOCK5_DEEP, BLOCK4_NUM, 2,training)
        print_activations(block5)
    # block to feature
    with tf.name_scope("block_feature"):

        batch_out = batch_norm(block5, training)
        print_activations(batch_out)

        pool_shape = batch_out.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        #reshaped = tf.reshape(pool3, [pool_shape[0], nodes])
        feature = tf.reshape(batch_out, [-1,1,1, nodes])
    #-------------------------------------------------------------------------
    # Gender Classifier
    # block gender classifier net layer 1
    with tf.variable_scope('block_gender_1'):
        gender_cnn_1 = fixed_conv2d(inputs=feature, filters=OUT1_DEEP, kernel_size=OUT1_SIZE, strides=1)
        gender_relu_1 = tf.nn.relu(gender_cnn_1)
        gender_block_1 = batch_norm(gender_relu_1, training)
        print_activations(gender_block_1)
    # block gender classifier net layer 2
    with tf.variable_scope('block_gender_2'):
        gender_cnn_2 = fixed_conv2d(inputs=gender_block_1, filters=OUT2_DEEP, kernel_size=OUT2_SIZE, strides=1)
        gender_relu_2 = tf.nn.relu(gender_cnn_2)
        gender_block_2 = batch_norm(gender_relu_2, training)
        print_activations(gender_block_2)
    # block gender classifier net layer 3
    with tf.variable_scope('block_gender_3'):
        gender_cnn_3 = fixed_conv2d(inputs=gender_block_2, filters=GENDER_CLASS_NUM, kernel_size=OUT3_SIZE, strides=1)
        gender_relu_3 = tf.nn.relu(gender_cnn_3)
        gender_block_3 = batch_norm(gender_relu_3, training)
        gender_out = tf.reshape(gender_block_3, [-1, GENDER_CLASS_NUM])
        print_activations(gender_out)
    #-------------------------------------------------------------------------
    # Age Classifier
    # block Age classifier net layer 1
    with tf.variable_scope('block_age_1'):
        age_cnn_1 = fixed_conv2d(inputs=feature, filters=OUT1_DEEP, kernel_size=OUT1_SIZE, strides=1)
        age_relu_1 = tf.nn.relu(age_cnn_1)
        age_block_1 = batch_norm(age_relu_1, training)
        print_activations(age_block_1)
    # block Age classifier net layer 2
    with tf.variable_scope('block_age_2'):
        age_cnn_2 = fixed_conv2d(inputs=age_block_1, filters=OUT2_DEEP, kernel_size=OUT2_SIZE, strides=1)
        age_relu_2 = tf.nn.relu(age_cnn_2)
        age_block_2 = batch_norm(age_relu_2, training)
        print_activations(age_block_2)
    # block Age classifier net layer 3
    with tf.variable_scope('block_age_3'):
        age_cnn_3 = fixed_conv2d(inputs=age_block_2, filters=AGE_CLASS_NUM, kernel_size=OUT3_SIZE, strides=1)
        age_relu_3 = tf.nn.relu(age_cnn_3)
        age_block_3 = batch_norm(age_relu_3, training)
        age_out = tf.reshape(age_block_3, [-1, AGE_CLASS_NUM])
        print_activations(age_out)


    return gender_out, age_out

