## experiments.py
# Experiments for CAProject

import os
import matplotlib.pyplot as plt
import time
import numpy as np
import datetime
from typing import List, Tuple
import tensorflow as tf

from model import CAModel
from train import train_ca
from image_processing import load_alive_image
from utils import manage_dir
from figures import FigGen

def is_model_trained(path, final_training_point):
  """Checks if a particular model has already been trained.
  Note that it does not account for different cell_fire_rate,
  or step_size."""
  
  if f'batches_{final_training_point}.jpg' in os.listdir(path+'/train_log'):
    print(f"Model already trained at {path}")
    try:
      with open(path+'loss_log.npy', 'rb') as file:
        return np.load(file)
    except FileNotFoundError: # In case model was accidentally trained without saving loss_log array
      print(f"WARNING: File {path+'loss_log.npy'} cannot be found")
      return []
  else:
    return []
  
class Experiments:
  def __init__(self, experiment_type, target_img, cell_fire_rate, step_size, hidden_size, channel_n, 
                target_padding, batch_size, pool_size, use_pattern_pool, damage_n, 
                steps, make_pool, path):
    """Note the init is passed in all the default params. It is up to the individual
    experiments to use which params they need and take as input anything else."""
    self.experiment_type = experiment_type
    self.target_img = target_img
    self.cell_fire_rate = cell_fire_rate
    self.step_size = step_size
    self.hidden_size = hidden_size
    self.channel_n = channel_n
    self.target_padding = target_padding
    self.batch_size = batch_size
    self.pool_size = pool_size
    self.use_pattern_pool = use_pattern_pool
    self.damage_n = damage_n
    self.steps = steps    
    self.make_pool = make_pool 
    self.path = path
  
  def experiment1(self, target_imgs: List[tuple], model_params: List[tuple]):
    """Experiment 1: Show the number of parameters needed for some target
    -log10 loss for multiple images.
    target_imgs: a list of tuples containing the name of the image and the target_img array
    model_params: a list of tuples containing the number of channels and hidden size.
      Corresponds to same index in image_names array. Note: len(image_names) must match len(model_params)ß
    """
    assert len(target_imgs) == len(model_params)
    # loss_log_dict = dict.fromkeys(image_names)
    loss_log_dict = {}

    # Work with each image individually to get the train_log array
    # for (image_name, max_size), (channel_n, hidden_size) in zip(image_names, model_params):
    for (image_name, target_img), (channel_n, hidden_size) in zip(target_imgs, model_params):
      # tf.keras.backend.clear_session()

      # Get loss_log 
      loss_log = is_model_trained(path, final_training_point=self.steps)
m
      if loss_log == []: # Model not previously trained
        path = f'figures/{image_name}/{self.experiment_type}/channel-{channel_n}_hidden-{hidden_size}'
        manage_dir(path, handle_train_log=True)
        print(f"\nRunning experiment 1 using image {image_name}.png")

        # Initialize model
        ca = CAModel(channel_n=channel_n, hidden_size=hidden_size, fire_rate=self.cell_fire_rate)
        ca.dmodel.summary()

        # Train it
        start_time = time.time()
        loss_log = train_ca(ca, target_img, channel_n, self.target_padding, self.batch_size,
                self.pool_size, self.use_pattern_pool, self.damage_n, steps=self.steps, path=path,
                make_pool=self.make_pool)
        print(f"\nTraining took {time.time() - start_time} seconds for image {image_name}.png")

        # Save training figures
        fig_gen = FigGen(ca, path)
        # steps = [100, 500, 800, 1000] if gpus else [0]
        fig_gen.training_progress_batches()

      # Append to big dictionary
      loss_log_dict[image_name] = loss_log

    # Make the plot
    # plt.plot(figsize=(10, 4))
    for image_name in loss_log_dict:
      # plt.bar(image_name, loss_log_dict[image_name][-1], '.', alpha=0.3)
      plt.plot(loss_log_dict[image_name], '.', alpha=0.3)
    plt.title(f'Loss history (log10) for {len(target_imgs)} images')

    # Save figure to current timestamp
    output_dir = f'figures/experiments/experiment1/'
    manage_dir(output_dir=output_dir)
    output_file = str(datetime.datetime.now())[:-7]
    plt.savefig(output_dir+output_file)

    



