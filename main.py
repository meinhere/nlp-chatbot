# IMPORT LIBRARY
import json
import nltk
import time
import random
import string
import pickle
import numpy as np
import pandas as pd
# from gtts import gTTS
from io import BytesIO
# import tensorflow as tf
import IPython.display as ipd
# import speech_recognition as sr
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
# from tensorflow.keras.models import Model
# from tensorflow.keras.utils import plot_model
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.layers import Input, Embedding, LSTM
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.layers import Flatten, Dense, GlobalMaxPool1D

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

with open('D:\Project\\02. python\\nlp\chatbot-uts\kampus_merdeka.json') as content:
  data1 = json.load(content)
# Mendapatkan semua data ke dalam list
tags = [] # data tag
inputs = [] # data input atau pattern
responses = {} # data respon
words = [] # Data kata
classes = [] # Data Kelas atau Tag
documents = [] # Data Kalimat Dokumen
responses = {}
ignore_words = ['?', '!']

for intent in data1['intents']:
  responses[intent['tag']]=intent['responses']
  for lines in intent['patterns']:
    inputs.append(lines)
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
      w = nltk.word_tokenize(pattern)
      words.extend(w)
      documents.append((w, intent['tag']))
      # add to our classes list
      if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Konversi data json ke dalam dataframe
data = pd.DataFrame({"patterns": inputs, "tags": tags})
