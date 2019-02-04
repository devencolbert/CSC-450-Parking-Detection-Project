import numpy as np
import cv2
import matplotlib.pyplot as plt
import importlib
import os, glob
import camera client

#moduleName = input('Enter module name: ')
#importlib.import_module(moduleName)

def all_images(images, cmap=None):
    cols = 3
    rows = (len(images)+1)//cols

    plt.figure(figsize=(15, 12))

    for i, image in enumerate(images):
        plt.subplot(rows, cols, i+1)
        cmap = 'gray' if len(image.shape)==2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()

images = [plt.imread(path) for path in glob.glob('*.jpg')]
all_images(images)
