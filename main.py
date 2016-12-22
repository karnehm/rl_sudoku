#!/usr/bin/python3

import logging
import sys
import tensorflow as tf

from agent import Agent
from environment import Environment

def main(_):
    with tf.Session() as sess:
        environment = Environment()
        agent = Agent(environment, sess)
        agent.train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1 and sys.argv[1] == 'p':
        with tf.Session() as sess:
            environment = Environment()
            agent = Agent(environment, sess)
            agent.play()
    else:
        tf.app.run()

