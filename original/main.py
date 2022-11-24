#@title Initialize Training { vertical-output: true}

print('\n...........................IN main.py...........................')

# Imports
# import os
# import io
# import PIL.ImageDraw
# import base64
# import zipfile
# import json
# import requests
# import numpy as np
# import matplotlib.pylab as pl
# import matplotlib.pyplot as plt
# import glob

import tensorflow as tf

# from IPython.display import Image, HTML, clear_output
# import tqdm

# https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['FFMPEG_BINARY'] = 'ffmpeg'
# import moviepy.editor as mvp
# from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
# clear_output()

from model import CAModel, load_emoji, to_rgb
from helpers import imshow, zoom
from train import train_ca

print('...........................FINISHED IMPORTS...........................')

# Choose target image
TARGET_EMOJI = "🦎" #@param {type:"string"}
target_img = load_emoji(TARGET_EMOJI)
print(f'target emoji is {TARGET_EMOJI}')
imshow(zoom(to_rgb(target_img), 2), fmt='png')
# print(f'target image is: {target_img}')

with tf.device('/gpu:0'):
    ca = CAModel()
    train_ca(ca, target_img=target_img, steps=300)

# if __name__ == '__main__':
    # main()
