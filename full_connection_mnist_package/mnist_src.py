import tensorflow as tf
from tensorflow_core.examples.tutorials.mnist import input_data

# MNIST Dataset params
INPUT_NODE = 784
OUTPUT_NODE = 10

# Neoral Net params
LAYER1_NODE = 500
# batch size
BATCH_SIZE = 100
# learning rate params
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
# regularization param
REGULARIZATION_RATE = 0.0001
# training steps
TRAINING_STEPS = 30000
# moving average func params
MOVING_AVERAGE_DECAY = 0.99

# Relu activate func
# 3 layers full connection net
def inference(input_tensor,avg_class,weights1,biase1,weights2,biase2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1) + biase1)
        return tf.matmul(layer1,weights2)+biase2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1)) + avg_class.average(biase1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biase2)

# training function
def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')

    # hidden layer params gen
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biase1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

    # ouput layer params gen
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biase2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    y = inference(x,None,weights1,biase1, weights2, biase2)

    # training rounds variable
    global_step = tf.Variable(0,trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # using moving average function
    average_y = inference(x,variable_averages,weights1, biase1, weights2, biase2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY)

    # optimize loss with gradient_descent_optimizer
    # feedback and update the moving average
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    train_op = tf.group(train_step,variable_averages_op)
    # the max of the line is the corresponding class of the prediction
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #training start
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        test_feed = {x:mnist.test.images,y_:mnist.test.labels}
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print("After %d training step(s),validation accuracy using average model is %g "%(i,validate_acc))
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("After %d training step(s),test accuracy using average model is %g " % (TRAINING_STEPS,test_acc))

def mnist_main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data",one_hot=True)
    train(mnist)