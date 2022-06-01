from skimage.util import random_noise
from skimage import io, filters
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def distort(img):
    """
    adds noise and blur to a segmented image to create fake raw data
    :img: a single image (np.ndarray)
    """
    img = img.astype(np.float32)
    img +=1.5
    img *= 1/np.amax(img)
    # img -= 0.1

    distorted = filters.gaussian(img, sigma=2)
    distorted = random_noise(distorted, mode='localvar', seed=3)
    # distorted = filters.gaussian(distorted, sigma=0.5)
    return distorted

def crop_save(img, size, path):
    'crops and saves images'
    new_img = img[:size, :size]
    plt.imsave(path, new_img, cmap='gray', vmin=0, vmax=1)

img = plt.imread('data/nmc_segmented.png')
distorted = distort(img)
crop_save(distorted, 512, 'data/synthetic_x.png')
crop_save(img, 512, 'data/synthetic_y.png')