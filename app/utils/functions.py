import io
import logging
import pdfplumber
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt

from . import llmParser
from InstructorEmbedding import INSTRUCTOR

def extractText(docFile):
    """ Extract the text from a PDF document given as UploadedFile or file-like object using pdfplumber.

        Parameters:
            - docFile (UploadedFile or file-like): PDF document in memory.

        Returns:
            - str: Extracted text from the PDF document.
    """
    with pdfplumber.open(docFile) as pdf:
        return "\n".join([page.extract_text() or '' for page in pdf.pages])


def extractCVsInfo(cvFiles):
    """ Extracts the information of the CVs given as a list of UploadedFile objects in memory.

        Parameters:
            - cvFiles (list of UploadedFile): List of CV PDFs loaded in memory.

        Returns:
            - cvDict (dict): Dictionary containing the extracted information of each CV, indexed by an integer key.
    """

    # Evitar warnings innecesarios de pdfplumber
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    logging.getLogger("pdfplumber").setLevel(logging.ERROR)

    parser = llmParser.LLMParser()
    cvDict = {}

    for i, file in enumerate(cvFiles):
        if file.name.endswith('.pdf'):
            text = extractText(file)  # Procesar PDF desde objeto en memoria
            cvInfo = parser.extract_CVInfo(text, True)
            cvDict[i] = cvInfo
            print(f"Processing: {file.name}")

    return cvDict


def extractJobDescriptionInfo(jobDescriptionFile):
    """ Extracts the information from a job description PDF file (UploadedFile) in memory.

        Parameters:
            - jobDescriptionFile (UploadedFile): PDF file uploaded in memory.

        Returns:
            - jobDescriptionInfo (dict): Extracted job description information.
    """

    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    logging.getLogger("pdfplumber").setLevel(logging.ERROR)

    parser = llmParser.LLMParser()
    jobDescriptionText = extractText(jobDescriptionFile)
    jobDescriptionInfo = parser.extract_jobDescriptionInfo(jobDescriptionText)

    return jobDescriptionInfo

def df2Excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='candidatos')
    
    output.seek(0)
    return output