import random

import tensorflow as tf
import numpy as np
from collections import deque

class DQN():
    def __init__(self,
                 state_dim=0,
                 action_dim=0,
                 gamma=0.9,
                 replay_size=10000,
                 batch_size = 32,
                 learning_rate = 0.0001,
                 hidden = 200 # hidden layer node number
                 ):
        self.time_step = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 0.5

        self.replay_buffer = deque()
        self.replay_size = replay_size
        self.batch_size = batch_size

        self.hidden = hidden
        self.lr = learning_rate

        self.create_neoral_nets()
        self.create_training_method()

        self.sess = tf.InteractiveSession()
        init = tf.initialize_all_variables()
        self.sess.run(init)

        # self.save_tf_graph()

    def create_neoral_nets(self):
        # online net
        with tf.variable_scope('online_net'):
            self.state = tf.placeholder(tf.float32,[None,self.state_dim],name='state')

            w1 = tf.Variable(tf.truncated_normal([self.state_dim,self.hidden],stddev=0.1))
            b1 = tf.Variable(tf.constant(0.01,shape=[self.hidden]))

            w2 = tf.Variable(tf.truncated_normal([self.hidden,self.action_dim],stddev=0.1))
            b2 = tf.Variable(tf.constant(0.01,shape=[self.action_dim]))

            layer1 = tf.nn.relu(
                tf.matmul(self.state,w1) + b1
            )

            self.q_eval = tf.matmul(layer1,w2) + b2

        # target net
        with tf.variable_scope('target_net'):
            self.next_state = tf.placeholder(tf.float32, [None, self.state_dim], name='next_state')

            w1_t = tf.Variable(tf.truncated_normal([self.state_dim, self.hidden]))
            b1_t = tf.Variable(tf.constant(0.01, shape=[self.hidden]))

            w2_t = tf.Variable(tf.truncated_normal([self.hidden, self.action_dim]))
            b2_t = tf.Variable(tf.constant(0.01, shape=[self.action_dim]))

            layer1_t = tf.nn.relu(
                tf.matmul(self.next_state, w1_t) + b1_t
            )

            self.q_target = tf.matmul(layer1_t, w2_t) + b2_t

    def create_training_method(self):
        self.y_input = tf.placeholder(tf.float32,[None],name='y_input')
        self.action_input = tf.placeholder(tf.float32,[None,self.action_dim],name='action_input')

        # batch-action Q_eval
        q_val = tf.reduce_sum(tf.multiply(self.q_eval,self.action_input),reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.y_input - q_val))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def save_tf_graph(self):
        tf.summary.FileWriter('log/',self.sess.graph)

    def choose_epsilon_action(self,state):
        if np.random.uniform() > self.epsilon:
            action_val = self.sess.run(self.q_eval,feed_dict={self.state:[state]})
            action = np.argmax(action_val)
        else:
            action = np.random.randint(0,self.action_dim)

        self.epsilon -= (0.5 - 0.01) / 10000
        return action

    def direct_action(self,state):
        action_val = self.sess.run(self.q_eval, feed_dict={self.state: [state]})
        return np.argmax(action_val)

    def train(self):
        if len(self.replay_buffer) > self.batch_size:
            self.neoral_net_training()
            # self.copy_params()

        self.time_step += 1


    def store_seq(self,state,action,reward,next_state,done):
        action_array = np.zeros(self.action_dim)

        action_array[action] = 1
        self.replay_buffer.append((state,action_array,reward,next_state,done))
        if len(self.replay_buffer) > self.replay_size:
            self.replay_buffer.popleft()


    def neoral_net_training(self):
        batch = random.sample(self.replay_buffer,self.batch_size)

        state_batch = [data[0] for data in batch]
        action_batch = [data[1] for data in batch]
        reward_batch = [data[2] for data in batch]
        next_state_batch = [data[3] for data in batch]

        q_target = self.q_eval.eval(feed_dict={self.next_state:next_state_batch})

        y_batch = []
        for i in range(self.batch_size):
            # done
            if batch[i][4]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.gamma * np.max(q_target[i]))

        _,self.cost = self.sess.run([self.train_op,self.loss],feed_dict={
            self.state:state_batch,
            self.action_input:action_batch,
            self.y_input:y_batch
        })
        # print(self.cost)


    def copy_params(self):
        t_params = tf.get_collection('target_net')
        e_params = tf.get_collection('online_net')

        self.replace_target_op = [tf.assign(t,e) for t,e in zip(t_params,e_params)]
        self.sess.run(self.replace_target_op)

    def save_model(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess,'../DQN_Models/DQN_final.ckpt')

class DDQN(DQN):
    def __init__(self,
                state_dim=0,
                action_dim=0
    ):
        super(DDQN,self).__init__(
                state_dim=state_dim,
                action_dim=action_dim,
                gamma=0.9,
                replay_size=10000,
                batch_size = 32,
                learning_rate = 0.0001,
                hidden = 200 # hidden layer node number
        )

    def neoral_net_training(self):
        batch = random.sample(self.replay_buffer,self.batch_size)

        state_batch = [data[0] for data in batch]
        action_batch = [data[1] for data in batch]
        reward_batch = [data[2] for data in batch]
        next_state_batch = [data[3] for data in batch]

        q_target = self.q_eval.eval(feed_dict={self.state:next_state_batch})
        q_eval = self.q_eval.eval(feed_dict={self.state:state_batch})

        y_batch = []
        for i in range(self.batch_size):
            # done
            if batch[i][4]:
                y_batch.append(reward_batch[i])
            else:
                idx = np.argmax(q_eval[i])
                y_batch.append(reward_batch[i] + self.gamma * q_target[i][idx])

        _,self.cost = self.sess.run([self.train_op,self.loss],feed_dict={
            self.state:state_batch,
            self.action_input:action_batch,
            self.y_input:y_batch
        })

    def save_model(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess,'../DQN_Models/DDQN_final.ckpt')
