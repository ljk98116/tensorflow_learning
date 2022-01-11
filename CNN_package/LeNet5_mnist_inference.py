import tensorflow as tf

# Neoral Net Params
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 1st Conv_Layer
CONV1_DEEP = 6
CONV1_SIZE = 5

# 2nd Conv_Layer
CONV2_DEEP = 16
CONV2_SIZE = 5

# Full Connection Layer
FC_SIZE1 = 120
FC_SIZE2 = 84

def inference(input_tensor,train,regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("biases",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(
            input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME'
        )
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(
            "weight",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("biases",[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(
            pool1,conv2_weights,strides=[1,1,1,1],padding='SAME'
        )
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        pool_shape = pool2.get_shape().as_list()
        print("pool shape:")
        print(pool_shape)
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2,[pool_shape[0],nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable(
            "weights",[nodes,FC_SIZE1],initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        fc1_biases = tf.get_variable(
            "biases",[FC_SIZE1],initializer=tf.constant_initializer(0.1)
        )
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1,0.5)
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable(
            "weights",[FC_SIZE1,FC_SIZE2],initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        fc2_biases = tf.get_variable(
            "biases",[FC_SIZE2],initializer=tf.constant_initializer(0.1)
        )
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2 = tf.nn.relu(tf.matmul(fc1,fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2,0.5)
    with tf.variable_scope('layer7-output'):
        output_weights = tf.get_variable(
            "weights",[FC_SIZE2,NUM_LABELS],initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        output_biases = tf.get_variable(
            "biases",[NUM_LABELS],initializer=tf.constant_initializer(0.1)
        )
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(output_weights))
        logit = tf.matmul(fc2,output_weights) + output_biases
    return logit