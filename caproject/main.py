#@title Initialize Training { vertical-output: true}

print('\n...........................IN main.py...........................')

# https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging
import tensorflow as tf
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import time
import zipfile

from image_processing import load_local_image, load_alive_image
from model import CAModel, load_emoji, to_rgb
from utils import imshow, zoom, export_ca_to_webgl_demo, manage_dir
from train import train_ca
from figures import FigGen
from experiments import Experiments

print('...........................FINISHED IMPORTS...........................')

def main():

    # Cellular Automata Parameters
    HIDDEN_SIZE = 128 # size of hidden layer in CNN
    CHANNEL_N = 16 # number of CA state channels
    TARGET_PADDING = 16 # number of pixels used to pad the target image border
    TARGET_SIZE = 125
    BATCH_SIZE = 8
    POOL_SIZE = 1024
    CELL_FIRE_RATE = 0.5
    STEP_SIZE = 1.0
    THRESHOLD = 0.01 # hyper-parameter denoting threshold for life

    EXPERIMENT_TYPE = "persistent" #@param ["Growing", "Persistent", "Regenerating"]
    EXPERIMENT_MAP = {"growing":0, "persistent":1, "regenerating":2}
    EXPERIMENT_N = EXPERIMENT_MAP[EXPERIMENT_TYPE]

    USE_PATTERN_POOL = [0, 1, 1][EXPERIMENT_N]
    DAMAGE_N = [0, 0, 3][EXPERIMENT_N]  # number of patterns to damage in a batch

    # Booleans to decide what information to store
    MAKE_CHECKPOINTS = False
    MAKE_POOL = False

    # Run experiments parameter
    EXPERIMENTS = [1, 0, 0, 0, 0]
    RUN_EXPERIMENTS = True

    # Do a clean run from scratch

    # Directory Management
    image_name = 'bob-ross-painting'
    output_dir = f'figures/{image_name}-/{EXPERIMENT_TYPE}/channel-{CHANNEL_N}_hidden-{HIDDEN_SIZE}'
    manage_dir(output_dir=output_dir+'/train_log', remove_flag=True)
    # try: # remove all files in train_log folder if it exists
    #     for file in os.listdir(output_dir+'/train_log/'):
    #         os.remove(output_dir+'/train_log/'+file)
    # except FileNotFoundError: # if it doesn't exist, create it
    #     os.makedirs(output_dir+'/train_log/')
    #     print(f"Created new directory to store output figures: {output_dir}")   
    # except OSError as e: # catch general OS errors
    #     print("Error: %s : %s" % (output_dir+'/train_log/', e.strerror))    

    # Select image to run on 
    # TARGET_EMOJI = '🛩'
    TARGET_EMOJI = None
    if TARGET_EMOJI != None:
        target_img = load_emoji(TARGET_EMOJI)
        print(f'target emoji is {TARGET_EMOJI}')
        imshow(zoom(to_rgb(target_img), 2), fmt='png', SAVE=True)
        # print(f'target image is: {target_img}')
    else: 
        target_img, _alpha_channel, _orig_img = load_alive_image(image_name, max_size=TARGET_SIZE)

        print(f"Using image {image_name}.png with max_size of {TARGET_SIZE}")

    # # Check for gpu
    # n_steps = 1000 # training steps
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #             logical_gpus = tf.config.list_logical_devices('GPU')
    #             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         print(e)
    # else:
    #     print("WARNING: Running without GPUs")
    #     n_steps = 1

    # Sort out which gpu to use
    with tf.device('/CPU:0'):
        physical_devices = tf.config.list_physical_devices('GPU') 
        gpus = (physical_devices != [])
        n_steps = 8000 # training steps
    if gpus:
        print("Num Physical GPUs Available:", len(tf.config.list_physical_devices('GPU')))
        # try:
        #     # Disable first GPU
        #     tf.config.set_visible_devices(physical_devices[1:], 'GPU')
        #     logical_devices = tf.config.list_logical_devices('GPU')
        #     print(f"Using {len(logical_devices)} logical gpus")
        #     # Logical device was not created for first GPU
        #     assert len(logical_devices) == len(physical_devices) - 1  
        # except:
        #     # Invalid device or cannot modify virtual devices once initialized.
        #     print("ERROR: Invalid device or cannot modify virtual devices once initialized.")
    else:
        print("WARNING: Running without GPUs")
        n_steps = 1

    if RUN_EXPERIMENTS:

        # target_img, _, _ = load_alive_image('bob-ross-painting', max_size=125)   
        # Run experiments from experiments module
        experiments = Experiments(EXPERIMENT_TYPE, target_img, CELL_FIRE_RATE, STEP_SIZE, 
                HIDDEN_SIZE, CHANNEL_N, TARGET_PADDING, BATCH_SIZE, POOL_SIZE, 
                USE_PATTERN_POOL, DAMAGE_N, n_steps, MAKE_POOL, output_dir)

        # Run first experiment
        image_names = ['bob-ross-painting', 'starry-night']
        target_img1, _, _ = load_alive_image(image_names[0], max_size=125)   
        target_img2, _, _ = load_alive_image(image_names[1], max_size=125)   
        target_imgs = [(image_names[0], target_img1), (image_names[1], target_img2)]
        model_params = [(20, 140), (20, 140)]
        experiments.experiment1(target_imgs, model_params)  

        return 0

    # Run the regular model
    # Initialize model
    ca = CAModel(channel_n=CHANNEL_N, hidden_size=HIDDEN_SIZE, fire_rate=CELL_FIRE_RATE)
    ca.dmodel.summary()

    # Train it
    start_time = time.time()
    _ = train_ca(ca, target_img, CHANNEL_N, TARGET_PADDING, BATCH_SIZE,
            POOL_SIZE, USE_PATTERN_POOL, DAMAGE_N, steps=n_steps, path=output_dir,
            make_pool=MAKE_POOL)
    print(f"\nTraining took {time.time() - start_time} seconds")

    # Save some figures of training progress
    fig_gen = FigGen(ca, output_dir)
    steps = [100, 500, 800, 1000] if gpus else [0]
    fig_gen.training_progress_batches()

    if MAKE_CHECKPOINTS:
        fig_gen.training_progress_checkpoints(damage_n=DAMAGE_N, channel_n=CHANNEL_N, steps=steps)

    if MAKE_POOL:
        fig_gen.pool_contents()

    # Export quantized model for WebGL demo
    if gpus:
        model = CAModel(channel_n=CHANNEL_N, hidden_size=HIDDEN_SIZE, fire_rate=CELL_FIRE_RATE)
        model.load_weights(output_dir+'/train_log/%04d'%n_steps)
        with zipfile.ZipFile(output_dir+'_webgl.zip', 'w') as zf:
            zf.writestr(f'channel-{CHANNEL_N}_hidden-{HIDDEN_SIZE}.json', export_ca_to_webgl_demo(model))

if __name__ == '__main__':
    main()
