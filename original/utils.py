print('\n...........................IN helpers.py...........................')

import io
import PIL.Image, PIL.ImageDraw
import base64
import numpy as np
from PIL import Image
import os
os.environ['FFMPEG_BINARY'] = 'ffmpeg'
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

def np2pil(a):
  if a.dtype in [np.float32, np.float64]:
    a = np.uint8(np.clip(a, 0, 1)*255)
  return PIL.Image.fromarray(a)

def imwrite(f, a, fmt=None):
  a = np.asarray(a)
  if isinstance(f, str):
    fmt = f.rsplit('.', 1)[-1].lower()
    if fmt == 'jpg':
      fmt = 'jpeg'
    f = open(f, 'wb')
  np2pil(a).save(f, fmt, quality=95)

def imencode(a, fmt='jpeg'):
  a = np.asarray(a)
  if len(a.shape) == 3 and a.shape[-1] == 4:
    fmt = 'png'
  f = io.BytesIO()
  imwrite(f, a, fmt)
  return f.getvalue()

def im2url(a, fmt='jpeg'):
  encoded = imencode(a, fmt)
  base64_byte_string = base64.b64encode(encoded).decode('ascii')
  return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string

def imshow(a, fmt='jpeg', SAVE=False):
  """Referenced: 
    https://stackoverflow.com/questions/60138697/typeerror-cannot-handle-this-data-type-1-1-3-f4
  """
  im = Image.fromarray((a*255).astype(np.uint8))
  im.show()
  # print(im)
  if SAVE:
    im.save(f'output/target_img.{fmt}')
#   display(Image(data=imencode(a, fmt)))

def tile2d(a, w=None):
  a = np.asarray(a)
  if w is None:
    w = int(np.ceil(np.sqrt(len(a))))
  th, tw = a.shape[1:3]
  pad = (w-len(a))%w
  a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant')
  h = len(a)//w
  a = a.reshape([h, w]+list(a.shape[1:]))
  a = np.rollaxis(a, 2, 1).reshape([th*h, tw*w]+list(a.shape[4:]))
  return a

def zoom(img, scale=4):
  img = np.repeat(img, scale, 0)
  img = np.repeat(img, scale, 1)
  return img


# Helper code to load image data
def load_alive_image(path, max_size, threshold, sigma=1.0):

  img = PIL.Image.open(path)
  img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
  img = np.float32(img)/255.0
  orig_img = img.copy()

  # Concatenate alpha channel initialized with zeros
  target_img = np.concatenate([img, np.ones_like(img[..., :1])], axis=-1)

  # Premultiply RGB by Alpha
  target_img[..., :3] *= target_img[..., 3:]

  return target_img, target_img[..., 3], orig_img
