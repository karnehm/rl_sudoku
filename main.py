#!/usr/bin/python3

import argparse
import logging

import tensorflow as tf

from agent import Agent
from environment import Environment


def main(_):
    with tf.Session() as sess:
        environment = Environment()
        agent = Agent(environment, sess)
        agent.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', type=int, nargs=1)
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.play:
        with tf.Session() as sess:
            environment = Environment()
            agent = Agent(environment, sess)
            agent.play(args.play[0])
    else:
        tf.app.run()