try:
    from PIL import Image
except ImportError:
    import Image
import pandas as pd
import numpy as np
import os
import pathlib
import glob
import cv2
import pytesseract
import re
from pdf2image import convert_from_bytes, convert_from_path
from pathlib import Path
from pdfquery import PDFQuery
from timeit import default_timer as timer

# define constants
PROJECT_DIR = "/Users/hammadsheikh/Desktop/Documents/Studies/Personal Projects/pdf_extractor/pdf_extractor/"
INPUT_DIR = PROJECT_DIR + 'input/'
OUTPUT_DIR = PROJECT_DIR + 'output/'

# FUNCTION DEFINITIONS

# Some help functions
# get average confidence value of OCR result
def get_conf(page_gray):
    df = pytesseract.image_to_data(page_gray,output_type = 'data.frame')
    df.drop(df[df.conf == -1].index.values, inplace = True)
    df.reset_index()
    return df.conf.mean()

# deskew the image
def deskew(image):
    gray = cv2.bitwise_not(image)
    temp_arr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(temp_arr > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)
    return rotated


"""
Main part of OCR:
pages_df: save extracted text for each pdf file, index by page
OCR_dic : dict for saving df of each pdf, filename is the key
"""

print("Starting PDF Extractor ...\n")
file_list = []

for file in glob.iglob(INPUT_DIR + '**/*.pdf', recursive = True):
    file_list.append(file)
    file_name = Path(file).name
    print("Processing file: " + file_name + " located at: " + os.path.dirname(file))
# %%time
OCR_dic={}
for file in file_list:
    # grab image name - we will need this later
    file_name = Path(file).name
    file_name, extension = os.path.splitext(file_name)

    # convert pdf into image
    pdf_file = convert_from_path(file)
    # create a df to save each pdf's text
    pages_df = pd.DataFrame(columns = ['conf','text'])
    for (i, page) in enumerate(pdf_file) :
        try:
            # transfer image of pdf_file into array
            page_arr = np.asarray(page)
            # transfer into grayscale
            page_arr_gray = cv2.cvtColor(page_arr,cv2.COLOR_BGR2GRAY)
            # deskew the page
            page_deskew = deskew(page_arr_gray)
            # cal confidence value
            page_conf = get_conf(page_deskew)
            # page text
            page_text = pytesseract.image_to_string(page_deskew)
            # extract string
            pages_df = pages_df._append({'conf': page_conf,'text': page_text}, ignore_index=True)
        except:
            # if can't extract then give some notes into df
            pages_df = pages_df._append({'conf': -1,'text': 'N/A'}, ignore_index = True)
            continue
        # save dataframe to text file
        print("File " + file_name + extension + " has been processed and is now being exported ...")
        pages_df[pages_df.columns[1]].to_csv(OUTPUT_DIR + file_name + '.txt', sep='\t', index = False)
        print("File " + file_name + ".txt has been created and saved!")
    # save df into a dict with filename as key
    OCR_dic[file] = pages_df
    # print('{} is done'.format(file))
