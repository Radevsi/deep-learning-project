## experiments.py
# Experiments for CAProject

import os
import sys
import matplotlib.pyplot as plt
import time
import numpy as np
import datetime
from typing import List
import tensorflow as tf
from keras.utils.layer_utils import count_params

from model import CAModel
from train import train_ca
from image_processing import load_alive_image, load_local_image
from utils import manage_dir
from figures import FigGen

def is_model_trained(path, final_training_point):
  """Checks if a particular model has already been trained.
  Note that it does not account for different cell_fire_rate,
  or step_size."""
  
  if f'batches_{final_training_point}.jpg' in os.listdir(path+'/train_log'):
    print(f"Model already trained at {path}")
    try:
      with open(path+'/loss_log.npy', 'rb') as file:
        return np.load(file)
    except FileNotFoundError: # In case model was accidentally trained without saving loss_log array
      print(f"WARNING: File {path+'loss_log.npy'} cannot be found")
      sys.exit()
  else:
    return []
  
class Experiments:
  def __init__(self, experiment_type, cell_fire_rate, step_size, hidden_size, channel_n, 
                target_padding, batch_size, pool_size, use_pattern_pool, damage_n, threshold, living_map,
                steps, make_pool):
    """Note the init is passed in all the default params. It is up to the individual
    experiments to use which params they need and take as input anything else."""
    self.experiment_type = experiment_type

    self.cell_fire_rate = cell_fire_rate
    self.step_size = step_size
    self.hidden_size = hidden_size
    self.channel_n = channel_n
    self.target_padding = target_padding
    self.batch_size = batch_size
    self.pool_size = pool_size
    self.use_pattern_pool = use_pattern_pool
    self.damage_n = damage_n
    self.threshold = threshold
    self.living_map = living_map
    self.steps = steps    
    self.make_pool = make_pool 
  
  def experiment1(self, image_names: List[str], target_sizes: List[int], model_params: List[tuple]):
    """Experiment 1: Show the number of parameters needed for some target
    -log10 loss for multiple images.
    image_names: a list of strings containing the name of the image
    target_sizes: a list of ints containing the target size of each corresponding image in target_imgs
    model_params: a list of tuples containing the number of channels and hidden size.
      Corresponds to same index in image_names array. Note: len(image_names) must equal len(model_params)
    """
    assert len(image_names) == len(model_params)
    loss_log_dict = {}

    # Initialize optimizer since it is a tf.Variable
    lr = 2e-3
    lr_sched=tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      [2000], [lr, lr*0.1])
    trainer=tf.keras.optimizers.Adam(lr_sched)

    # Define the figure
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))

    # Work with each image individually to get the train_log array
    for image_name, target_size, (channel_n, hidden_size) in zip(image_names, target_sizes, model_params):

      # File management
      path = f'figures/{image_name}-{target_size}/{self.experiment_type}/channel-{channel_n}_hidden-{hidden_size}'
      manage_dir(path+'/train_log', remove_flag=False)
      print(f"\nRunning experiment 1 using image {image_name}.png with target size of {target_size}")

      # Initialize the model
      ca = CAModel(channel_n=channel_n, hidden_size=hidden_size, fire_rate=self.cell_fire_rate)
      ca.dmodel.summary()
      n_model_params = count_params(ca.trainable_weights)

      # Get loss_log 
      loss_log = is_model_trained(path, final_training_point=self.steps)
      if loss_log == []: # model not previously trained

        # print("EXITING PREMATURELY")
        # sys.exit()
        
        # Get target_img array
        target_img = None
        if self.living_map[image_name]: # if image is alive
          target_img, _, _ = load_alive_image(image_name, max_size=target_size)
        else: # append alpha channels otherwise
          target_img, _, _ = load_local_image(image_name, max_size=target_size, threshold=self.threshold)

        # Train it
        start_time = time.time()
        loss_log = train_ca(ca, target_img, channel_n, self.target_padding, self.batch_size,
                self.pool_size, self.use_pattern_pool, self.damage_n, trainer=trainer, steps=self.steps, 
                path=path, make_pool=self.make_pool)
        print(f"\nTraining took {time.time() - start_time} seconds for image {image_name}.png")

        # Save training figures
        fig_gen = FigGen(ca, path)
        fig_gen.training_progress_batches()

        # Clear session and then delete model as per https://github.com/keras-team/keras/issues/5345
        tf.keras.backend.clear_session()
      del ca

      # Append to big dictionary
      loss_log_dict[image_name] = loss_log

      # Add the plots to the respective axes
      log10_loss_log = np.log10(loss_log)
      ax0.plot(log10_loss_log, '.', alpha=0.2, label=f'{image_name}-{target_size}')
      ax1.bar(f'{image_name}-{target_size}', n_model_params, alpha=0.5, label=f'Log10 Loss: {log10_loss_log[-1]:.2f}')

    # Style the figure
    ax0.set_title(f'Loss History (log10) for {len(image_names)} Images')
    ax0.set_xlabel('Number of Training Steps')
    ax0.set_ylabel('Negative Log10 Loss')
    ax0.legend()

    ax1.set_title(f'Number of Model Parameters Used For Each Image')
    ax1.set_ylabel(f'Number of Model Parameters')
    ax1.legend()

    # Save figure to current timestamp
    output_dir = f'figures/experiments/experiment1/'
    manage_dir(output_dir=output_dir)
    output_file = str(datetime.datetime.now())[:-7]
    fig.savefig(output_dir+output_file)


  def experiment2(self, image_name, target_size, model_params: List[tuple]):
    """
    target_img: The target_img array
    model_params: a list of tuples containing the number of channels and hidden size.
      Corresponds to same index in image_names array. Note: len(image_names) must equal len(model_params)
    """

    loss_log_dict = {}

    # Initialize optimizer since it is a tf.Variable
    lr = 2e-3
    lr_sched=tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      [2000], [lr, lr*0.1])
    trainer=tf.keras.optimizers.Adam(lr_sched)

    # Define the figure
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))

    # Go through all parameters that are asked to be plotted
    for (channel_n, hidden_size) in model_params:

      # File management
      path = f'figures/{image_name}-{target_size}/{self.experiment_type}/channel-{channel_n}_hidden-{hidden_size}'
      manage_dir(path+'/train_log', remove_flag=False)
      print(f"\nRunning experiment 1 using image {image_name}.png with target size of {target_size}")

      # Initialize the model
      ca = CAModel(channel_n=channel_n, hidden_size=hidden_size, fire_rate=self.cell_fire_rate)
      ca.dmodel.summary()
      n_model_params = count_params(ca.trainable_weights)

      # Get loss_log 
      loss_log = is_model_trained(path, final_training_point=self.steps)
      if loss_log == []: # model not previously trained

        # print("EXITING PREMATURELY")
        # sys.exit()
        
        # Get target_img array
        target_img = None
        if self.living_map[image_name]: # if image is alive
          target_img, _, _ = load_alive_image(image_name, max_size=target_size)
        else: # append alpha channels otherwise
          target_img, _, _ = load_local_image(image_name, max_size=target_size, threshold=self.threshold)

        # Train it
        start_time = time.time()
        loss_log = train_ca(ca, target_img, channel_n, self.target_padding, self.batch_size,
                self.pool_size, self.use_pattern_pool, self.damage_n, trainer=trainer, steps=self.steps, 
                path=path, make_pool=self.make_pool)
        print(f"\nTraining took {time.time() - start_time} seconds for image {image_name}.png")

        # Save training figures
        fig_gen = FigGen(ca, path)
        fig_gen.training_progress_batches()

        # Clear session and then delete model as per https://github.com/keras-team/keras/issues/5345
        tf.keras.backend.clear_session()
      del ca

      # Append to big dictionary
      loss_log_dict[image_name] = loss_log

      # Add the plots to the respective axes
      log10_loss_log = np.log10(loss_log)
      ax0.plot(log10_loss_log, '.', alpha=0.2, label=f'{image_name}-{target_size}')
      ax1.bar(f'{image_name}-{target_size}', n_model_params, alpha=0.5, label=f'Log10 Loss: {log10_loss_log[-1]:.2f}')

    # Style the figure
    ax0.set_title(f'Loss History (log10) for {len(image_names)} Images')
    ax0.set_xlabel('Number of Training Steps')
    ax0.set_ylabel('Negative Log10 Loss')
    ax0.legend()

    ax1.set_title(f'Number of Model Parameters Used For Each Image')
    ax1.set_ylabel(f'Number of Model Parameters')
    ax1.legend()

    # Save figure to current timestamp
    output_dir = f'figures/experiments/experiment1/'
    manage_dir(output_dir=output_dir)
    output_file = str(datetime.datetime.now())[:-7]
    fig.savefig(output_dir+output_file)

    


  # def experiment3(self, cell_fire_rates: List[float]):
  #   """Experiment 3: Change the cell_fire_rate parameter and see results"""

    
    


  #   for cell_fire_rate in cell_fire_rates:
  #     path = f'figures/experiments/experiment3/{self.experiment_type}/channel-{self.channel_n}_hidden-{self.hidden_size}_fr-{cell_fire_rate}'
  #     ca = CAModel(channel_n=self.channel_n, hidden_size=self.hidden_size, fire_rate=cell_fire_rate)
  #     ca.dmodel.summary()

  #   # Train it
  #   start_time = time.time()
  #   loss_log = train_ca(ca, self.target_img, self.channel_n, self.target_padding, self.batch_size,
  #           self.pool_size, self.use_pattern_pool, self.damage_n, steps=self.steps, path=path,
  #           make_pool=self.make_pool)
  #   print(f"\nTraining took {time.time() - start_time} seconds")      


    



