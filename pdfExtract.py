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
from timeit import default_timer as timer
from PyPDF2 import PdfReader
from pytesseract import Output

# define constants
PROJECT_DIR = "/Users/hammadsheikh/Desktop/Documents/Studies/Personal Projects/pdf_extractor/pdf_extractor/"
INPUT_DIR = PROJECT_DIR + 'input/'
OUTPUT_DIR = PROJECT_DIR + 'output/'

# define variables
# dict for saving df of each pdf, filename is the key
OCR_dic = {}

# FUNCTION DEFINITIONS

# help functions
# fix orientation for text searchable pdf
def fix_txt_rot(page):
    OrientationDegrees = page.get('/Rotate')
    if OrientationDegrees is None:
        OrientationDegrees = 0
    while OrientationDegrees > 0:
        page.rotate(90)
        OrientationDegrees = (page.get('/Rotate') % 360)
    return page

# extract text from text searchable pdf
def text_pdf(file):
    # creating a pdf reader object
    reader = PdfReader(file)
    # store number of pages in pdf file, to use in for loop
    no_pages = len(reader.pages)
    # extracting text from each page
    extracted_text = ""
    for i in range(no_pages):
        page = reader.pages[i]
        page = fix_txt_rot(page)
        extracted_text += page.extract_text()
    return extracted_text

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

# OCR function
# pages_df: dataframe to save extracted text for each pdf file, index by page
def ocr(file):
    # convert pdf into image
    pdf_file = convert_from_path(file)
    # create a df to save each pdf's text
    pages_df = pd.DataFrame(columns = ['conf','text'])
    for (i, page) in enumerate(pdf_file):
        try:
            # get image orientation and fix it if needed
            osd_data = pytesseract.image_to_osd(page, output_type = Output.DICT)
            page_rotation = osd_data['rotate']
            if page_rotation > 0:
                page_rotated = page.rotate(-page_rotation)
            else:
                page_rotated = page
            # transfer image of pdf_file into array
            page_arr = np.asarray(page_rotated)
            # transfer into grayscale
            page_arr_gray = cv2.cvtColor(page_arr,cv2.COLOR_BGR2GRAY)
            # deskew the page
            page_deskew = deskew(page_arr_gray)
            # cal confidence value
            page_conf = get_conf(page_deskew)
            # page text
            page_text = pytesseract.image_to_string(page_deskew)
            # extract string
            pages_df = pages_df._append({'conf': page_conf,'text': page_text}, ignore_index = True)
        except:
            # if can't extract then give some notes into df
            pages_df = pages_df._append({'conf': -1,'text': 'N/A'}, ignore_index = True)
            continue
    return pages_df

print("Starting PDF Extractor ...\n")

for file in glob.iglob(INPUT_DIR + '**/*.pdf', recursive = True):
    file_name = Path(file).name
    file_name, extension = os.path.splitext(file_name)
    print("Processing file: " + file_name + " located at: " + os.path.dirname(file))

    # try to extract text from pdf without OCR
    text = text_pdf(file)

    if len(text) < 10:
        # if we weren't able to extract text without OCR, we try OCR
        pages_df = pd.DataFrame(columns = ['conf','text'])
        pages_df = ocr(file)
        # save dataframe to text file
        print("File " + file_name + extension + " has been processed via OCR and is now being exported ...")
        pages_df[pages_df.columns[1]].to_csv(OUTPUT_DIR + file_name + '.txt', sep = '\t', index = False)
        print("File " + file_name + ".txt has been created and saved!")
        # save df into a dict with filename as key
        OCR_dic[file] = pages_df
    else:
        print("File " + file_name + extension + " has been processed via text parser and is now being exported ...")
        # save the extracted text
        output_file = os.path.join(OUTPUT_DIR, file_name + ".txt")
        out_file = open(output_file, "w")
        for t in text:
            out_file.write(t)
        out_file.close()
        print("File " + file_name + ".txt has been created and saved!")
