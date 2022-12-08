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
from utils import imshow, zoom, export_ca_to_webgl_demo
from train import train_ca
from figures import FigGen
import time
import zipfile

print('...........................FINISHED IMPORTS...........................')

def main():

    # Cellular Automata Parameters
    HIDDEN_SIZE = 128 # size of hidden layer in CNN
    CHANNEL_N = 20 # number of CA state channels
    TARGET_PADDING = 16 # number of pixels used to pad the target image border
    TARGET_SIZE = 125
    BATCH_SIZE = 8
    POOL_SIZE = 1024
    CELL_FIRE_RATE = 0.5

    THRESHOLD = 0.01 # hyper-parameter denoting threshold for life

    EXPERIMENT_TYPE = "persistent" #@param ["Growing", "Persistent", "Regenerating"]
    EXPERIMENT_MAP = {"growing":0, "persistent":1, "regenerating":2}
    EXPERIMENT_N = EXPERIMENT_MAP[EXPERIMENT_TYPE]

    USE_PATTERN_POOL = [0, 1, 1][EXPERIMENT_N]
    DAMAGE_N = [0, 0, 3][EXPERIMENT_N]  # number of patterns to damage in a batch

    # Booleans to decided what information to store
    MAKE_CHECKPOINTS = False
    MAKE_POOL = False

    # Pathing variables
    image_name = 'bob-ross-painting'
    output_dir = f'figures/{image_name}/{EXPERIMENT_TYPE}/channel-{CHANNEL_N}_hidden-{HIDDEN_SIZE}'
    try: # remove all files in train_log folder if it exists
        for file in os.listdir(output_dir+'/train_log/'):
            os.remove(output_dir+'/train_log/'+file)
    except FileNotFoundError: # if it doesn't exist, create it
        os.makedirs(output_dir+'/train_log/')
        print(f"Created new directory to store output figures: {output_dir}")   
    except OSError as e: # catch general OS errors
        print("Error: %s : %s" % (output_dir+'/train_log/', e.strerror))    

    # Select image to run on 
    # TARGET_EMOJI = '🛩'
    TARGET_EMOJI = None
    if TARGET_EMOJI != None:
        target_img = load_emoji(TARGET_EMOJI)
        print(f'target emoji is {TARGET_EMOJI}')
        imshow(zoom(to_rgb(target_img), 2), fmt='png', SAVE=True)
        # print(f'target image is: {target_img}')
    else: 
        load_path = f'images/{image_name}.png'
        target_img, _alpha_channel, _orig_img = load_alive_image(load_path, max_size=TARGET_SIZE)

    print(f"Using image {image_name}.png with max_size of {TARGET_SIZE}")

    # Initialize model
    ca = CAModel(channel_n=CHANNEL_N, hidden_size=HIDDEN_SIZE, fire_rate=CELL_FIRE_RATE)
    ca.dmodel.summary()

    # Check for gpu
    gpu = (tf.config.list_physical_devices('GPU') != [])
    n_steps = 1000 # training steps
    if gpu:
        print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    else:
        print("WARNING: Running without GPUs")
        n_steps = 1

    # Train it
    start_time = time.time()
    train_ca(ca, target_img, CHANNEL_N, TARGET_PADDING, BATCH_SIZE,
             POOL_SIZE, USE_PATTERN_POOL, DAMAGE_N, steps=n_steps, path=output_dir,
             make_pool=MAKE_POOL)
    print(f"\nTraining took {time.time() - start_time} seconds")

    # Save some figures of training progress
    fig_gen = FigGen(ca, output_dir)
    steps = [100, 500, 800, 1000] if gpu else [0]
    fig_gen.training_progress_batches()

    if MAKE_CHECKPOINTS:
        fig_gen.training_progress_checkpoints(damage_n=DAMAGE_N, channel_n=CHANNEL_N, steps=steps)

    if MAKE_POOL:
        fig_gen.pool_contents()

    # Export quantized model for WebGL demo
    if gpu:
        model = CAModel(channel_n=CHANNEL_N, hidden_size=HIDDEN_SIZE, fire_rate=CELL_FIRE_RATE)
        model.load_weights(output_dir+'/train_log/%04d'%n_steps)
        with zipfile.ZipFile(output_dir+'webgl_models8.zip', 'w') as zf:
            zf.writestr(f'channel-{CHANNEL_N}_hidden-{HIDDEN_SIZE}.json', export_ca_to_webgl_demo(model))

if __name__ == '__main__':
    main()
