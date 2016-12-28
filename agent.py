"""
Agent implementation.
"""

import logging
import math
import random

import numpy as np
import tensorflow as tf

import sudoku

SUDOKU_SIZE = 9


class Agent:
    def __init__(self, env, sess):
        """
        Agent object, which learns to solve sudoku grids via Deep Q-learning.

        :param env: Environment for Agent to interact with
        :param sess: TensorFlow session
        """
        self.step = 0
        self.num_episodes = int(1e5)

        self.env = env
        self.sess = sess
        self.history = []

        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_time = 0.3 * self.num_episodes

        self.discount = 0.99

        self.play_mode = False

        self._setup_dqn()
        self._restore()

    def choose_action(self):
        epsilon = (self.epsilon_start +
                   (min(self.epsilon_time, self.step) / self.epsilon_time) *
                   (self.epsilon_end - self.epsilon_start))

        if not self.play_mode and random.random() < epsilon:
            action = random.randrange(self.env.num_actions)
        else:
            action = self.max_q_action.eval(
                {self.state: [self.env.current_grid]})[0]

        return action

    def act(self, action):
        logging.debug("Taking action %d", action)
        return self.env.act(action)

    def _sample_history(self):
        sample_indices = np.random.randint(0, len(self.history), 32)
        sample = [self.history[i] for i in sample_indices]

        grids = np.vstack([grid for grid, _,  _, _, _ in sample])
        actions = np.array([action for _, action, _, _, _ in sample])
        rewards = np.array([reward for _, _, reward, _, _ in sample])
        terminals = np.array([terminal for _, _, _, terminal, _ in sample])
        next_grids = np.vstack([next_grid for _, _, _, _, next_grid in sample])

        return grids, actions, rewards, terminals, next_grids

    def _do_q_learning(self):
        logging.debug("Q-learning phase")
        if len(self.history) < 32:
            return

        if len(self.history) > 50000:
            sample_indices = np.random.randint(0, len(self.history), 40000)
            self.history = [self.history[i] for i in sample_indices]
        grid, action, reward, terminal, next_grid = self._sample_history()

        # Use target network to predict expected reward
        q_action_next = self.target_max_q.eval({self.state: next_grid})
        terminal = np.array(terminal)
        y = reward + (1. - terminal) * self.discount * q_action_next

        _, q_t, loss = self.sess.run(
            [self.optim, self.q, self.loss],
            {
                self.y: y,
                self.action: action,
                self.state: grid,
            })

        logging.debug("Loss is %s", loss)

    def _update_target(self):
        for key in self.w:
            tf.assign(self.target_w[key], self.w[key])
        for key in self.b:
            tf.assign(self.target_b[key], self.b[key])

    def _play_once(self, train_mode=True, display=False):
        grid = self.env.new_grid()
        if display:
            print(sudoku.unflatten(grid))

        terminal = False
        game_length = 0
        while not terminal:
            action = self.choose_action()
            new_grid, reward, terminal = self.act(action)

            if train_mode:
                self.step += 1
                new_grid = np.zeros(SUDOKU_SIZE ** 3) if new_grid is None else new_grid
                self.history.append(
                    (grid.copy(), action, reward, terminal, new_grid))

                self._do_q_learning()
                if self.step % 100 == 0:
                    self._update_target()

            if display:
                print("Taking action {}... Reward: {}, Terminal {}".format(
                    action, reward, terminal))
                print(sudoku.unflatten(new_grid))

            game_length += 1
            grid = new_grid

        return int(reward > 0), game_length

    def play(self, num_games=1, display=False):
        """
        Play a number of sudoku grids according to the currently-learnt policy.

        :param num_games: number fof sudoku grids to attempt to solve
        :param display: whether the individual moves hould be printed
        :return: None
        """
        self.play_mode = True
        num_successes = 0
        total_length = 0
        if self.restored:
            for _ in range(num_games):
                success, game_length = self._play_once(train_mode=False, display=display)
                num_successes += success
                total_length += game_length

            print("Played {} games. Successes: {}, average game length: {}".format(
                num_games, num_successes, total_length / num_games))

    def train(self):
        """
        Train the Agent via Deep Q-learning, and store learnt weights.

        :return: None
        """
        try:
            game_successes = []
            game_lengths = []
            summary_successes = []
            summary_lengths = []

            for i in range(self.num_episodes):
                success, game_length = self._play_once(train_mode=True)
                game_successes.append(success)
                game_lengths.append(game_length)

                if i % 100 == 0:
                    summary_successes.append(np.sum(game_successes))
                    summary_lengths.append(np.mean(game_lengths))
                    game_successes = []
                    game_lengths = []
                    logging.info("Game: %s, average length: %s, number of successes: %s",
                                 i, summary_lengths[-1], summary_successes[-1])

        except KeyboardInterrupt:
            pass

        self.saver.save(self.sess, 'data/model.ckpt')
        np.savetxt('data/successes.txt', summary_successes)
        np.savetxt('data/lengths.txt', summary_lengths)

    def _setup_dqn(self):
        SQUARE_SIDE = int(math.sqrt(SUDOKU_SIZE))
        CONV_WINDOWS = SUDOKU_SIZE
        NUM_ACTIONS = SUDOKU_SIZE ** 3

        def conv2d(x, W, strides=[1, 1, 1, 1]):
            return tf.nn.conv2d(x, W, strides=strides, padding='VALID')

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.5, shape=shape)
            return tf.Variable(initial)

        self.w = {}
        self.b = {}
        self.target_w = {}
        self.target_b = {}

        # Input state
        self.state = tf.placeholder('float32', [None, SUDOKU_SIZE ** 3])
        x = tf.reshape(self.state, [-1, SUDOKU_SIZE, SUDOKU_SIZE ** 2, 1])

        # Convolution over one-hot encoding of individual grid entries
        self.w['entry'] = weight_variable((1, SUDOKU_SIZE, 1, CONV_WINDOWS))
        self.b['entry'] = bias_variable((CONV_WINDOWS,))
        h_conv1 = tf.nn.relu(conv2d(x, self.w['entry'], strides=[1, 1, SUDOKU_SIZE, 1]) + self.b['entry'])

        self.target_w['entry'] = weight_variable((1, SUDOKU_SIZE, 1, CONV_WINDOWS))
        self.target_b['entry'] = bias_variable((CONV_WINDOWS,))
        target_h_conv1 = tf.nn.relu(
            conv2d(x, self.target_w['entry'], strides=[1, 1, SUDOKU_SIZE, 1]) + self.target_b['entry'])

        # Convolution over rows, columns, and boxes
        self.w['row'] = weight_variable((1, SUDOKU_SIZE, CONV_WINDOWS, CONV_WINDOWS ** 2))
        self.b['row'] = bias_variable((CONV_WINDOWS ** 2,))
        self.w['col'] = weight_variable((SUDOKU_SIZE, 1, CONV_WINDOWS, CONV_WINDOWS ** 2))
        self.b['col'] = bias_variable((CONV_WINDOWS ** 2,))
        self.w['box'] = weight_variable((SQUARE_SIDE, SQUARE_SIDE, CONV_WINDOWS, CONV_WINDOWS ** 2))
        self.b['box'] = bias_variable((CONV_WINDOWS ** 2,))

        h_row = tf.nn.relu(conv2d(h_conv1, self.w['row'], strides=[1, 1, SUDOKU_SIZE, 1]) + self.b['row'])
        h_col = tf.nn.relu(conv2d(h_conv1, self.w['col'], strides=[1, SUDOKU_SIZE, 1, 1]) + self.b['col'])
        h_box = tf.nn.relu(conv2d(h_conv1, self.w['box'], strides=[1, SQUARE_SIDE, SQUARE_SIDE, 1]) + self.b['box'])

        h_row_flat = tf.reshape(h_row, [-1, SUDOKU_SIZE * CONV_WINDOWS ** 2])
        h_col_flat = tf.reshape(h_col, [-1, SUDOKU_SIZE * CONV_WINDOWS ** 2])
        h_box_flat = tf.reshape(h_box, [-1, SUDOKU_SIZE * CONV_WINDOWS ** 2])
        h_all = tf.concat(1, [h_row_flat, h_col_flat, h_box_flat])

        self.target_w['row'] = weight_variable((1, SUDOKU_SIZE, CONV_WINDOWS, CONV_WINDOWS ** 2))
        self.target_b['row'] = bias_variable((CONV_WINDOWS ** 2,))
        self.target_w['col'] = weight_variable((SUDOKU_SIZE, 1, CONV_WINDOWS, CONV_WINDOWS ** 2))
        self.target_b['col'] = bias_variable((CONV_WINDOWS ** 2,))
        self.target_w['box'] = weight_variable((SQUARE_SIDE, SQUARE_SIDE, CONV_WINDOWS, CONV_WINDOWS ** 2))
        self.target_b['box'] = bias_variable((CONV_WINDOWS ** 2,))

        target_h_row = tf.nn.relu(
            conv2d(target_h_conv1, self.w['row'], strides=[1, 1, SUDOKU_SIZE, 1]) + self.b['row'])
        target_h_col = tf.nn.relu(
            conv2d(target_h_conv1, self.w['col'], strides=[1, SUDOKU_SIZE, 1, 1]) + self.b['col'])
        target_h_box = tf.nn.relu(
            conv2d(target_h_conv1, self.w['box'], strides=[1, SQUARE_SIDE, SQUARE_SIDE, 1]) + self.b['box'])

        target_h_row_flat = tf.reshape(target_h_row, [-1, SUDOKU_SIZE * CONV_WINDOWS ** 2])
        target_h_col_flat = tf.reshape(target_h_col, [-1, SUDOKU_SIZE * CONV_WINDOWS ** 2])
        target_h_box_flat = tf.reshape(target_h_box, [-1, SUDOKU_SIZE * CONV_WINDOWS ** 2])
        target_h_all = tf.concat(1, [target_h_row_flat, target_h_col_flat, target_h_box_flat])

        # Final, fully-connected layer
        self.w['final'] = weight_variable([3 * SUDOKU_SIZE * CONV_WINDOWS ** 2, NUM_ACTIONS])
        self.b['final'] = bias_variable([NUM_ACTIONS])
        self.q = tf.matmul(h_all, self.w['final']) + self.b['final']

        self.target_w['final'] = weight_variable([3 * SUDOKU_SIZE * CONV_WINDOWS ** 2, NUM_ACTIONS])
        self.target_b['final'] = bias_variable([NUM_ACTIONS])
        self.target_q = tf.matmul(target_h_all, self.target_w['final']) + self.target_b['final']

        # Calculate loss
        self.max_q_action = tf.argmax(self.q, dimension=1)
        # self.max_q = tf.reduce_max(self.q, 1)

        # self.target_max_q_action = tf.argmax(self.target_q, dimension=1)
        self.target_max_q = tf.reduce_max(self.target_q, 1)

        self.y = tf.placeholder('float32', [None], name='y')
        self.action = tf.placeholder('int64', [None], name='action')

        action_one_hot = tf.one_hot(self.action, self.env.num_actions)
        q_with_action = tf.reduce_sum(
            self.q * action_one_hot, reduction_indices=1, name='q_acted')

        self.loss = tf.reduce_mean(tf.square(self.y - q_with_action), name='loss')

        # Update weights
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optim = tf.train.RMSPropOptimizer(
            learning_rate=0.001, momentum=0.95, epsilon=0.01).minimize(
            self.loss, global_step=global_step)

        tf.initialize_all_variables().run()

        self.saver = tf.train.Saver(list(self.w.values()) + list(self.b.values()))

    def _restore(self):
        ckpt = tf.train.get_checkpoint_state('data')
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.restored = True
