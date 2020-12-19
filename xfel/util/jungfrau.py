from __future__ import absolute_import,print_function, division
import numpy as np


def correct_panel(img, copy=True):
  """
  Distributes the intensity in the larger Jungfrau pixels into smaller
  inserted pixels
  See: https://doi.org/10.1088/1748-0221/13/11/C11006


  Parameters
  ==========
  img: a 2D numpy of shape 512x1024

  copy: boolean, if True, copy the image, otherwise
  the input image is updated in-place (usually not desired).
  The performance hit for copy=True is negligible in most applications.
  On psana servers this function runs in ~6 ms with copy=False
  and ~7.5 ms with copy=True

  Return
  ======
  2D numpy array of shape 514x1030
  """

  if not isinstance(img, np.ndarray):
    raise TypeError("input image needs to be a numpy array")
  if img.shape != (512, 1024):
    raise ValueError("Input image needs shape 512x1024")

  if copy:
    img = img.copy()

  img[255]/=2
  img[256]/=2
  img2 = np.insert(img, (256, 256), values=(img[255], img[256]), axis=0).T

  img2[255]/=2
  img2[256]/=2
  img2[511]/=2
  img2[512]/=2
  img2[767]/=2
  img2[768]/=2

  img3 = np.insert(img2, (256, 256, 512, 512, 768, 768),
    values=(img2[255], img2[256],
            img2[511], img2[512],
            img2[767], img2[768]),
    axis=0).T
  return img3
