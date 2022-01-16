# test LSTM for sin function regression
from RNN_learning_package.sin_function_regression import *

test_start = (TRAINING_EXAMPLES + TRAINING_STEPS) * SAMPLE_GAP
test_end = test_start + (TRAINING_EXAMPLES + TRAINING_STEPS) * SAMPLE_GAP
train_X,train_y = generate_data(np.sin(np.linspace(0,test_start,TRAINING_EXAMPLES + TIMESTEPS,dtype=np.float32)))
test_X,test_y = generate_data(np.sin(np.linspace(test_start,test_end,TESTING_EXAMPLES + TIMESTEPS,dtype=np.float32)))

with tf.Session() as sess:
    train(sess,train_X,train_y)
    run_eval(sess,test_X,test_y)