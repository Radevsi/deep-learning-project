#@title Initialize Training { vertical-output: true}

print('\n...........................IN main.py...........................')

# https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging
import tensorflow as tf
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

from image_processing import load_local_image, load_alive_image
from model import CAModel, load_emoji, to_rgb
from utils import imshow, zoom
from train import train_ca
from figures import FigGen
import time

print('...........................FINISHED IMPORTS...........................')

def main():

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices != []:
        print("Num GPUs Available:", len(physical_devices))
    else:
        print("WARNING: Running without GPUs")

    # Cellular Automata Parameters
    HIDDEN_SIZE = 128 # size of hidden layer in CNN
    CHANNEL_N = 22 # number of CA state channels
    TARGET_PADDING = 16 # number of pixels used to pad the target image border
    TARGET_SIZE = 125
    BATCH_SIZE = 8
    POOL_SIZE = 1024
    CELL_FIRE_RATE = 0.5

    THRESHOLD = 0.01

    EXPERIMENT_TYPE = "Persistent" #@param ["Growing", "Persistent", "Regenerating"]
    EXPERIMENT_MAP = {"Growing":0, "Persistent":1, "Regenerating":2}
    EXPERIMENT_N = EXPERIMENT_MAP[EXPERIMENT_TYPE]

    USE_PATTERN_POOL = [0, 1, 1][EXPERIMENT_N]
    DAMAGE_N = [0, 0, 3][EXPERIMENT_N]  # number of patterns to damage in a batch

    # Select image to run on 
    # TARGET_EMOJI = '🛩'
    TARGET_EMOJI = None
    if TARGET_EMOJI != None:
        target_img = load_emoji(TARGET_EMOJI)
        print(f'target emoji is {TARGET_EMOJI}')
        imshow(zoom(to_rgb(target_img), 2), fmt='png', SAVE=True)
        # print(f'target image is: {target_img}')
    else: 
        load_path = 'images/bob-ross-painting.png'
        target_img, _alpha_channel, _orig_img = load_alive_image(load_path, max_size=TARGET_SIZE)

    # Initialize model and train it

    # with tf.device('/gpu:0'):
    ca = CAModel(channel_n=CHANNEL_N, hidden_size=HIDDEN_SIZE, fire_rate=CELL_FIRE_RATE)
    ca.dmodel.summary()

    start_time = time.time()
    train_ca(ca, target_img, CHANNEL_N, TARGET_PADDING, BATCH_SIZE,
             POOL_SIZE, USE_PATTERN_POOL, DAMAGE_N, steps=1000)
    print(f"\nTraining took {time.time() - start_time} seconds")

    # Save some figures of training progress
    fig_gen = FigGen(ca)
    # steps = [100, 500, 1000, 4000]
    steps = [100, 500]
    fig_gen.training_progress_checkpoints(damage_n=DAMAGE_N, channel_n=CHANNEL_N, steps=steps)
    fig_gen.training_progress_batches()
    fig_gen.pool_contents()


if __name__ == '__main__':
    main()
