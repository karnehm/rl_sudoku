"""
Agent implementation.
"""

import logging
import random

import numpy as np
import tensorflow as tf

import environment
import sudoku


NUM_FEATURES = 4
SUDOKU_SIZE = 4
NUM_ACTIONS = 4*4*4


def conv2d(x, W, strides=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='VALID')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.5, shape=shape)
    return tf.Variable(initial)


class Agent:
    def __init__(self, environment, sess):
        self.step = 0
        self.num_episodes = 10000
        self.episode_length = 10000
        self.gradient_update_step = 10
        
        self.env = environment
        self.sess = sess
        self.history = []

        self.epsilon_start = 1.0
        self.epsilon_end = 0.2
        self.epsilon_time = 100000

        self.discount = 0.99

        self.play_mode = False

        self.setup_dqn()

    def choose_action(self):
        epsilon = (self.epsilon_start + 
            (min(self.epsilon_time, self.step)/self.epsilon_time) * 
            (self.epsilon_end - self.epsilon_start))

        if not self.play_mode and random.random() < epsilon:
            action = random.randrange(self.env.num_actions)
        else:
            action = self.max_q_action.eval(
                {self.state: [self.env.current_grid]})[0]

        return action

    def act(self, action):
        return self.env.act(action)

#    def observe(self):
#        if self.step % self.episode_length == 0:
#            self.do_q_learning()
#
#        if self.step % self.gradient_update_step == 0:
#            self.update_target()

    def sample_history(self):
        np.random.shuffle(self.history)
        sample = self.history[:32]
        grids = np.vstack([grid for grid,_,_,_,_ in sample])
        actions = np.array([action for _,action,_,_,_ in sample])
        rewards = np.array([reward for _,_,reward,_,_ in sample])
        terminals = np.array([terminal for _,_,_,terminal,_ in sample])
        next_grids = np.vstack([next_grid for _,_,_,_,next_grid in sample])
        return grids, actions, rewards, terminals, next_grids
    
    def do_q_learning(self):
        logging.debug("Q-learning phase")
        if len(self.history) < 32: return
        if len(self.history) > 10000:
            self.history = self.history[1000:]
        grid, action, reward, terminal, next_grid = self.sample_history()
        
        q_action_next = self.target_max_q.eval({self.state: next_grid})

        terminal = np.array(terminal) + 0.
        y = reward + (1. - terminal) * self.discount * q_action_next

        _, q_t, loss = self.sess.run(
            [self.optim, self.q, self.loss],
            {
                self.y: y,
                self.action: action,
                self.state: grid,
            })
        logging.debug("Loss is %s", loss)

    def update_target(self):
        for key in self.w:
            tf.assign(self.target_w[key], self.w[key])
        for key in self.b:
            tf.assign(self.target_b[key], self.b[key])

    def play(self):
        #import pdb; pdb.set_trace()
        self.play_mode = True
        ckpt = tf.train.get_checkpoint_state('.')
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

            grid = self.env.new_grid()
            print(sudoku.unflatten(grid))
            terminal = False
            while not terminal:
                action = self.choose_action()
                logging.info("Taking action %d", action)
                new_grid, reward, terminal = self.act(action)
                logging.info("Reward: %d, Terminal %d", reward, terminal)

                print(sudoku.unflatten(new_grid))
                grid = new_grid

    def train(self, n_iters=1000):
        try:
            for i in range(self.num_episodes):
                self.train_episode(i)
        except KeyboardInterrupt:
            pass

        self.saver.save(self.sess, './model.ckpt')

    def train_episode(self, i):
        logging.info("Episode %d", i)
        game_lengths = []
        game_length = 0
        grid = self.env.new_grid()
        num_victories = 0
        ep_start = self.step
        while num_victories == 0:
            action = self.choose_action()
            logging.debug("Taking action %d", action)
            new_grid, reward, terminal = self.act(action)
            game_length += 1
            logging.debug("Reward: %d, Terminal %d", reward, terminal)

            if terminal:
                new_grid = np.zeros(64)
                
            self.history.append(
                (grid.copy(), action, reward, terminal, new_grid))
            if terminal:
                grid = self.env.reset_grid()
                game_lengths.append(game_length)
                game_length = 0
                if reward > 0:
                    num_victories += 1
            else:
                grid = new_grid

            self.step += 1

            self.do_q_learning()
            if self.step % 100 == 0:
                self.update_target()

            if self.step % 1000 == 0:
                logging.info("Step %s, Average game length %s",
                    self.step, np.mean(game_lengths))

        logging.info("Episode length %s", self.step - ep_start)
        logging.info("Average game length %s", np.mean(game_lengths))
        self.history = []

    def setup_dqn(self):
        self.w = {}
        self.b = {}
        self.target_w = {}
        self.target_b = {}

        # Input state
        self.state = tf.placeholder('float32', [None, SUDOKU_SIZE**3])
        x = tf.reshape(self.state, [-1, SUDOKU_SIZE, SUDOKU_SIZE**2, 1])

        self.w['entry'] = weight_variable((1, SUDOKU_SIZE, 1, SUDOKU_SIZE))
        self.b['entry'] = bias_variable((SUDOKU_SIZE,))
        h_conv1 = tf.nn.relu(conv2d(x, self.w['entry'], strides=[1, 1, SUDOKU_SIZE, 1]) + self.b['entry'])

        self.target_w['entry'] = weight_variable((1, SUDOKU_SIZE, 1, SUDOKU_SIZE))
        self.target_b['entry'] = bias_variable((SUDOKU_SIZE,))
        target_h_conv1 = tf.nn.relu(
            conv2d(x, self.target_w['entry'], strides=[1, 1, SUDOKU_SIZE, 1]) + self.target_b['entry'])
        
        # Convolution over rows, columns, and boxes
        self.w['row'] = weight_variable((1, 4, SUDOKU_SIZE, SUDOKU_SIZE**2))
        self.b['row'] = bias_variable((SUDOKU_SIZE**2,))
        self.w['col'] = weight_variable((4, 1, SUDOKU_SIZE, SUDOKU_SIZE**2))
        self.b['col'] = bias_variable((SUDOKU_SIZE**2,))
        self.w['box'] = weight_variable((2, 2, SUDOKU_SIZE, SUDOKU_SIZE**2))
        self.b['box'] = bias_variable((SUDOKU_SIZE**2,))

        h_row = tf.nn.relu(conv2d(h_conv1, self.w['row'], strides=[1, 1, 4, 1]) + self.b['row'])
        h_col = tf.nn.relu(conv2d(h_conv1, self.w['col'], strides=[1, 4, 1, 1]) + self.b['col'])
        h_box = tf.nn.relu(conv2d(h_conv1, self.w['box'], strides=[1, 2, 2, 1]) + self.b['box'])

        h_row_flat = tf.reshape(h_row, [-1, 4*SUDOKU_SIZE**2])
        h_col_flat = tf.reshape(h_col, [-1, 4*SUDOKU_SIZE**2])
        h_box_flat = tf.reshape(h_box, [-1, 4*SUDOKU_SIZE**2])

        h_all = tf.concat(1, [h_row_flat, h_col_flat, h_box_flat])

        self.target_w['row'] = weight_variable((1, 4, SUDOKU_SIZE,SUDOKU_SIZE**2))
        self.target_b['row'] = bias_variable((SUDOKU_SIZE**2,))
        self.target_w['col'] = weight_variable((4, 1, SUDOKU_SIZE,SUDOKU_SIZE**2))
        self.target_b['col'] = bias_variable((SUDOKU_SIZE**2,))
        self.target_w['box'] = weight_variable((2, 2, SUDOKU_SIZE,SUDOKU_SIZE**2))
        self.target_b['box'] = bias_variable((SUDOKU_SIZE**2,))

        target_h_row = tf.nn.relu(conv2d(target_h_conv1, self.w['row'], strides=[1, 1, 4, 1]) + self.b['row'])
        target_h_col = tf.nn.relu(conv2d(target_h_conv1, self.w['col'], strides=[1, 4, 1, 1]) + self.b['col'])
        target_h_box = tf.nn.relu(conv2d(target_h_conv1, self.w['box'], strides=[1, 2, 2, 1]) + self.b['box'])

        target_h_row_flat = tf.reshape(target_h_row, [-1, 4*SUDOKU_SIZE**2])
        target_h_col_flat = tf.reshape(target_h_col, [-1, 4*SUDOKU_SIZE**2])
        target_h_box_flat = tf.reshape(target_h_box, [-1, 4*SUDOKU_SIZE**2])

        target_h_all = tf.concat(1, [target_h_row_flat, target_h_col_flat, target_h_box_flat])

        # Final
        self.w['final'] = weight_variable([4*3*SUDOKU_SIZE**2, 64])
        self.b['final'] = bias_variable([64])

        self.q = tf.matmul(h_all, self.w['final']) + self.b['final']
#       self.q = tf.nn.dropout(
#            (tf.matmul(h_all, self.w['final']) + self.b['final']),
#            0.9)
        
        self.target_w['final'] = weight_variable([4*3*SUDOKU_SIZE**2, 64])
        self.target_b['final'] = bias_variable([64])

        self.target_q = tf.matmul(target_h_all, self.target_w['final']) + self.target_b['final']
        
        # Calculate loss
        self.max_q_action = tf.argmax(self.q, dimension=1)
        self.max_q = tf.reduce_max(self.q, 1)
        
        self.target_max_q_action = tf.argmax(self.target_q, dimension=1)
        self.target_max_q = tf.reduce_max(self.target_q, 1)

        self.y = tf.placeholder('float32', [None], name='y')
        self.action = tf.placeholder('int64', [None], name='action')

        action_one_hot = tf.one_hot(self.action, self.env.num_actions)
        q_with_action = tf.reduce_sum(
            self.q * action_one_hot, reduction_indices=1, name='q_acted')

        self.delta = self.y - q_with_action

        self.loss = tf.reduce_mean(tf.square(self.delta), name='loss')

        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optim = tf.train.RMSPropOptimizer(
            learning_rate=0.001, momentum=0.95, epsilon=0.01).minimize(
                self.loss, global_step=global_step)

        tf.initialize_all_variables().run()

        self.saver = tf.train.Saver(list(self.w.values()) + list(self.b.values()))
