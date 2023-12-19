import pandas as pd
import numpy as np
import os
import pathlib
import glob
from pathlib import Path
from pdfquery import PDFQuery
from timeit import default_timer as timer

# define constants
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
    file_name = Path(file_path).name
    file_name, extension = os.path.splitext(file_name)

    # this is only for console logging
    print("Processing PDF file: " + file_name + " ... ")

    # load pdf file
    pdf = PDFQuery(file_path)
    pdf.load()

    # Use CSS-like selectors to locate the elements
    text_elements = pdf.pq('LTTextLineHorizontal')

    # Extract the text from the elements
    text = [t.text for t in text_elements]

    print(text)

    # save the text extracted
    output_file = os.path.join(OUTPUT_DIR, file_name + ".txt")

    out_file = open(output_file, "w")
    for t in text:
        out_file.write(t)
    out_file.close()

    # set end timer
    end = timer()
    print("PDF file " + file_name + " processed in: " + str(int(end - start)) + " seconds")

# recursively load and process each image
for filename in glob.iglob(INPUT_DIR + '**/*.pdf', recursive = True):
    process_pdf(filename)
