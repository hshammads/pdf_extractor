import pandas as pd
import numpy as np
import os
import PIL
import PIL.Image as pilim
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt
import cv2
import pathlib
import glob
from tensorflow import keras
from pathlib import Path
from timeit import default_timer as timer

# define constants
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
PROJECT_DIR = "/Users/hammadsheikh/Desktop/Documents/Studies/Personal Projects/pdf_extractor/pdf_extractor/"
INPUT_DIR = PROJECT_DIR + 'input/'
OUTPUT_DIR = PROJECT_DIR + 'output/'

# FUNCTION DEFINITIONS

# define parsing and preprocessing function for the images
# the following function
def process_pdf(file_path):

    # set start timer
    start = timer()

    # grab image name - we will need this later
    image_name = Path(file_path).name

    # this is only for console logging
    print("Processing PDF file: " + image_name + " ... ")

    # read original image
    original_image = cv2.imread(file_path)

    # copy image to HSV
    img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # convert image to 2D array
    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized)

    # use k-means to segment the image
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    attempts = 5
    ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    # get the segmentated image and resize it
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    # result_image = cv2.resize(result_image, IMG_SIZE, interpolation = cv2.INTER_AREA)

    # save the segmented image to drive
    file_save_path = OUTPUT_DIR + image_name
    plt.imsave(file_save_path, result_image)

    # set end timer
    end = timer()
    print("PDF file " + image_name + " processed in: " + str(int(end - start)) + " seconds")

# recursively load and process each image
for filename in glob.iglob(INPUT_DIR + '**/*.pdf', recursive=True):
    process_pdf(filename)
