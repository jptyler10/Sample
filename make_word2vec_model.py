import random
from gensim.models import Word2Vec
import pandas as pd
import json
import re
import string
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from utils import clean_text, text2words


def word2vec_model(df, vector_size=50):
    
    print('Cleaning the text...\n\n')

    df['cleaned'] = df.note_text.apply(lambda x: clean_text(x))
    
    df['text'] = df.cleaned.str.split(r'\s+')
    
    texts = df['text'].tolist()
    
    print('Running the Word2Vec Model...\n\n')
    
    model = Word2Vec(sentences=texts, size=vector_size, min_count=1)
    
    return model

