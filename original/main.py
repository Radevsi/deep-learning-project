#@title Initialize Training { vertical-output: true}

print('\n...........................IN main.py...........................')

# https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging
import tensorflow as tf
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

from model import CAModel, load_emoji, to_rgb
from utils import imshow, zoom, load_alive_image
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

    # Choose target image
    CHANNEL_N = 22
    # TARGET_EMOJI = "🦎" #@param {type:"string"}
    TARGET_EMOJI = '🛩'
    target_img = load_emoji(TARGET_EMOJI)
    print(f'target emoji is {TARGET_EMOJI}')
    imshow(zoom(to_rgb(target_img), 2), fmt='png', SAVE=True)
    # print(f'target image is: {target_img}')

    EXPERIMENT_TYPE = "Persistent" #@param ["Growing", "Persistent", "Regenerating"]
    EXPERIMENT_MAP = {"Growing":0, "Persistent":1, "Regenerating":2}
    EXPERIMENT_N = EXPERIMENT_MAP[EXPERIMENT_TYPE]
    THRESHOLD = 0.01

    USE_PATTERN_POOL = [0, 1, 1][EXPERIMENT_N]
    DAMAGE_N = [0, 0, 3][EXPERIMENT_N]  # Number of patterns to damage in a batch

    load_path = 'images/bob-ross-painting.png'
    target_img, alpha_channel, orig_img = load_alive_image(load_path, max_size=125, 
        threshold=THRESHOLD)

    # print(f'The image (shape: {orig_img.shape}) we are working with is: ')
    # plt.imshow(orig_img)
    # plt.show()

    # with tf.device('/gpu:0'):
    ca = CAModel(channel_n=CHANNEL_N)
    start_time = time.time()
    train_ca(ca, target_img=target_img, 
            use_pattern_pool=USE_PATTERN_POOL, 
            damage_n=DAMAGE_N, channel_n=CHANNEL_N,
            steps=8000)

    print(f"\nTraining took {time.time() - start_time} seconds")

    # Save some figures of training progress
    fig_gen = FigGen(ca)
    steps = [100, 500, 1000, 4000]
    # steps = [100]
    fig_gen.training_progress_checkpoints(damage_n=DAMAGE_N, channel_n=CHANNEL_N, steps=steps)
    fig_gen.training_progress_batches()
    fig_gen.pool_contents()


if __name__ == '__main__':
    main()
