import pandas as pd 
import numpy as np

import string 
import re #regex library

# import word_tokenize & FreqDist from NLTK
import nltk
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from PyPDF2 import PdfReader
from nltk.corpus import stopwords


# ------ Tokenizing ---------
def word_page_count(text, search):
    list_page = []
    pages = text.split("\f")
    for i, page_text in enumerate(pages):
        if re.findall(search, page_text):
            count_page = len(re.findall(search, page_text))
            list_page.append((count_page, i + 1))

    count = len(list_page)
    total = sum([tup[0] for tup in list_page])

    return total, count

def save_to_txt(file_name, text):
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(text)

# Extract Text From PDF File
def extract_text_from_pdf(file):
    pdf_text = ""
    pdf_reader = PdfReader(file)
    num_pages = len(pdf_reader.pages)

    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        pdf_text += page.extract_text()

    return pdf_text

def remove_name_and_date(extracted_text):
     # Remove the numbering
    extracted_text = re.sub(r'#\s*\d+\s*', '', extracted_text, flags=re.I)
    # Remove the time and date
    extracted_text = re.sub(r'\s*(Senin|Selasa|Rabu|Kamis|Jum\'?at|Sabtu|Minggu)\s*,\s*\d{1,2}\s*(Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s*\d{4}\s*\d{1,2}:\d{1,2}(:\d{1,2})?\s*(WIB)?', '', extracted_text)
    # Remove the name pattern, which is any text followed by parentheses
    extracted_text = re.sub(r'.*?\(.*?\).*', '', extracted_text)
    # Remove the pattern "Forum Penelaahan Pagu Anggaran" and any lines following it up to "Unit :"
    extracted_text = re.sub(r'Forum Penelaahan Pagu Anggaran', '', extracted_text)
    # Remove the pattern "Unit:" and any lines following it up to "Unit :"
    extracted_text = re.sub(r'Unit\s*:.*?(?=\n\n|$)', '', extracted_text)
    # Remove the "Unit :" pattern and anything that follows until the end of the line
    extracted_text = re.sub(r'Unit\s*:\s*\d+\.\d+\s.*', '', extracted_text)
    # Remove the "ID PENELAAHAN" pattern
    extracted_text = re.sub(r'PENELAAHAN', '', extracted_text)
    # Remove the "ID PENELAAHAN" pattern
    extracted_text = re.sub(r'ID', '', extracted_text)
    # Remove multiple spaces and newlines
    extracted_text = re.sub(r'\s{2,}', ' ', extracted_text).strip()
    extracted_text = extracted_text.strip()
    return extracted_text

def preprocess(extracted_text):
    extracted_text = extracted_text.replace("\n", " ")
    return extracted_text

def preprocess2(text):
    # Case Folding
    text = text.lower()
    # Remove leading and trailing whitespaces
    text = text.strip()
    # # Remove multiple whitespaces into a single whitespace
    text = re.sub('\s+', ' ', text)
    # # Remove single characters
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    ## Remove number
    text = re.sub(r"\d+", "", text)
    # # Tokenization
    text = word_tokenize(text)
    # # Remove stopwords
    list_stopwords = stopwords.words('indonesian')
    list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang', 'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 'jd', 'jgn', 'sdh', 'aja', 'n', 't','tks', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt', '&amp', 'yah','wkwk',"assalamu'alaikum",'wr','wb','bapak','ibu', 'wabillahi','taufik','wal','hidayah',"wassalamu'alaikum", 'yth', 'puji','syukur','hadirat','allah','swt','karunia', 'selamat','pagi','siang','approval','close','forum','atau', 'halaman','yang','dan','terima','kasih','mohon','terimakasih','bapakibu','bpkibu','dengan','maaf','bapak','ibu'])
    txt_stopword = pd.read_csv("stopword.txt", names=["stopword"], header=None)
    list_stopwords.extend(txt_stopword["stopword"][0].split(' '))
    text = [word for word in text if word not in list_stopwords]
    # Normalization
    normalized_word = pd.read_excel("normalisasi-V1.xlsx")
    normalized_word_dict = dict(zip(normalized_word['slang'], normalized_word['formal']))
    text = [normalized_word_dict[word] if word in normalized_word_dict else word for word in text] 
    # Delete punctuation
    punctuations = '''!()-[]{};:'"\,<>@#$%^&*_~'''
    text = " ".join(text)
    text = ''.join(char for char in text if char not in punctuations)
    
    return text
