#!/usr/bin/python3

import logging
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
    tf.app.run()
