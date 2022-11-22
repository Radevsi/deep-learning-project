# Module for training the CA model

import os
import io
import PIL.Image, PIL.ImageDraw
import base64
import zipfile
import json
import requests
import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import glob

import tensorflow as tf

from IPython.display import Image, HTML, clear_output
import tqdm

import os

from helpers import imshow, imwrite, tile2d
os.environ['FFMPEG_BINARY'] = 'ffmpeg'
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
clear_output()

# Initialize training
from model import TARGET_PADDING, CAModel, load_emoji, to_rgb, to_rgba

from google.protobuf.json_format import MessageToDict
from tensorflow.python.framework import convert_to_constants

# TARGET_EMOJI = "🦎" #@param {type:"string"}


# Train Utilities (SamplePool, Model Export, Damage)
class SamplePool:
  def __init__(self, *, _parent=None, _parent_idx=None, **slots):
    self._parent = _parent
    self._parent_idx = _parent_idx
    self._slot_names = slots.keys()
    self._size = None
    for k, v in slots.items():
      if self._size is None:
        self._size = len(v)
      assert self._size == len(v)
      setattr(self, k, np.asarray(v))

  def sample(self, n):
    idx = np.random.choice(self._size, n, False)
    batch = {k: getattr(self, k)[idx] for k in self._slot_names}
    batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
    return batch

  def commit(self):
    for k in self._slot_names:
      getattr(self._parent, k)[self._parent_idx] = getattr(self, k)

@tf.function
def make_circle_masks(n, h, w):
  x = tf.linspace(-1.0, 1.0, w)[None, None, :]
  y = tf.linspace(-1.0, 1.0, h)[None, :, None]
  center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5)
  r = tf.random.uniform([n, 1, 1], 0.1, 0.4)
  x, y = (x-center[0])/r, (y-center[1])/r
  mask = tf.cast(x*x+y*y < 1.0, tf.float32)
  return mask

def export_model(ca, base_fn):
  ca.save_weights(base_fn)

  cf = ca.call.get_concrete_function(
      x=tf.TensorSpec([None, None, None, CHANNEL_N]),
      fire_rate=tf.constant(0.5),
      angle=tf.constant(0.0),
      step_size=tf.constant(1.0))
  cf = convert_to_constants.convert_variables_to_constants_v2(cf)
  graph_def = cf.graph.as_graph_def()
  graph_json = MessageToDict(graph_def)
  graph_json['versions'] = dict(producer='1.14', minConsumer='1.14')
  model_json = {
      'format': 'graph-model',
      'modelTopology': graph_json,
      'weightsManifest': [],
  }
  with open(base_fn+'.json', 'w') as f:
    json.dump(model_json, f)

def generate_pool_figures(pool, step_i):
  tiled_pool = tile2d(to_rgb(pool.x[:49]))
  fade = np.linspace(1.0, 0.0, 72)
  ones = np.ones(72) 
  tiled_pool[:, :72] += (-tiled_pool[:, :72] + ones[None, :, None]) * fade[None, :, None] 
  tiled_pool[:, -72:] += (-tiled_pool[:, -72:] + ones[None, :, None]) * fade[None, ::-1, None]
  tiled_pool[:72, :] += (-tiled_pool[:72, :] + ones[:, None, None]) * fade[:, None, None]
  tiled_pool[-72:, :] += (-tiled_pool[-72:, :] + ones[:, None, None]) * fade[::-1, None, None]
  imwrite('train_log/%04d_pool.jpg'%step_i, tiled_pool)

def visualize_batch(x0, x, step_i):
  vis0 = np.hstack(to_rgb(x0).numpy())
  vis1 = np.hstack(to_rgb(x).numpy())
  vis = np.vstack([vis0, vis1])
  imwrite('train_log/batches_%04d.jpg'%step_i, vis)
  print('batch (before/after):')
  imshow(vis)

def plot_loss(loss_log):
  pl.figure(figsize=(10, 4))
  pl.title('Loss history (log10)')
  pl.plot(np.log10(loss_log), '.', alpha=0.1)
  # pl.show()
  pl.draw()


# target_img = load_emoji(TARGET_EMOJI)
# p=TARGET_PADDING
# pad_target = tf.pad(target_img, [(p, p), (p, p), (0, 0)])

def loss_f(x, pad_target):
  return tf.reduce_mean(tf.square(to_rgba(x)-pad_target), [-2, -3, -1])

@tf.function
def train_step(x, trainer, pad_target, ca=CAModel()):
  """Applies single training step. Unchanged."""
  # lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
  #   [2000], [lr, lr*0.1])
  # trainer = tf.keras.optimizers.Adam(lr_sched)
  iter_n = tf.random.uniform([], 64, 96, tf.int32)
  with tf.GradientTape() as g:
    for i in tf.range(iter_n):
      x = ca(x)
    loss = tf.reduce_mean(loss_f(x, pad_target))
  grads = g.gradient(loss, ca.weights)
  grads = [g/(tf.norm(g)+1e-8) for g in grads]
  trainer.apply_gradients(zip(grads, ca.weights))
  return x, loss

from model import CHANNEL_N, POOL_SIZE
BATCH_SIZE = 8
EXPERIMENT_TYPE = "Regenerating" #@param ["Growing", "Persistent", "Regenerating"]
EXPERIMENT_MAP = {"Growing":0, "Persistent":1, "Regenerating":2}
EXPERIMENT_N = EXPERIMENT_MAP[EXPERIMENT_TYPE]

USE_PATTERN_POOL = [0, 1, 1][EXPERIMENT_N]
DAMAGE_N = [0, 0, 3][EXPERIMENT_N]  # Number of patterns to damage in a batch

def train_ca(ca, target_img, steps=8000, p=TARGET_PADDING, lr=2e-3):
  """
  Main training function. 
  Equivalent to 'Training Loop' in Colab Notebook.
  """
  print('TRAINING...')

  lr_sched=tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [2000], [lr, lr*0.1])
  trainer=tf.keras.optimizers.Adam(lr_sched)

  loss_log = []
  pad_target = tf.pad(target_img, [(p, p), (p, p), (0, 0)])
  h, w = pad_target.shape[:2]
  seed = np.zeros([h, w, CHANNEL_N], np.float32)
  seed[h//2, w//2, 3:] = 1.0

  # loss0 = loss_f(seed).numpy()
  pool = SamplePool(x=np.repeat(seed[None, ...], POOL_SIZE, 0))

  for i in range(steps+1):
    print(f'At step {i} in training loop')
    if USE_PATTERN_POOL:
      batch = pool.sample(BATCH_SIZE)
      x0 = batch.x
      loss_rank = loss_f(x0, pad_target).numpy().argsort()[::-1]
      x0 = x0[loss_rank]
      x0[:1] = seed
      if DAMAGE_N:
        damage = 1.0-make_circle_masks(DAMAGE_N, h, w).numpy()[..., None]
        x0[-DAMAGE_N:] *= damage
    else:
      x0 = np.repeat(seed[None, ...], BATCH_SIZE, 0)

    x, loss = train_step(x0, trainer, pad_target, ca=ca)
    print('Exited train_step')

    if USE_PATTERN_POOL:
      batch.x[:] = x
      batch.commit()

    step_i = len(loss_log)
    loss_log.append(loss.numpy())
    print('appended to loss_log')
    
    if step_i%10 == 0:
      print('generating pool figures...')
      generate_pool_figures(pool, step_i)
      print('...finished generating pool figures')
    if step_i%100 == 0:
      print('starting to clear output...')
      clear_output()
      print('cleared output...')
      visualize_batch(x0, x, step_i)
      print('visualizing batch...')
      plot_loss(loss_log)
      print('plotted loss...')
      export_model(ca, 'train_log/%04d'%step_i)
      print('exported model...')

    print('\r step: %d, log10(loss): %.3f'%(len(loss_log), np.log10(loss)), end='')


# ca = CAModel()

# !mkdir -p train_log && rm -f train_log/*