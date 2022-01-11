import time
import tensorflow as tf
from tensorflow_core.examples.tutorials.mnist import input_data
from CNN_package.LeNet5_mnist_inference import *
from CNN_package.LeNet5_mnist_train import *

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as q:
        print(mnist.validation.images.shape[0])
        x = tf.placeholder(tf.float32,[mnist.validation.images.shape[0],IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS],name='x-input')
        y_ = tf.placeholder(tf.float32,[None,NUM_LABELS],name='y-input')
        reshaped_x = np.reshape(mnist.validation.images,(mnist.validation.images.shape[0],IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS))
        validate_feed = {x:reshaped_x,y_:mnist.validation.labels}
        y = inference(x,False,None)

        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # load model
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    # iterate round number
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
                    print("After %s training step(s),validation accuracy = %g"%(global_step,accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("/path/to/mnist_data",one_hot=True)
    evaluate(mnist)

if __name__ == "__main__":
    tf.app.run()