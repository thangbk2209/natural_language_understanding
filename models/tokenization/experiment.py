from tokenizer import Tokenizer
from data_cleaner import DataCleaner
import pandas as pd 
import numpy as np
import pickle as pk

corpus_file = '../../data/corpus.txt'
file_to_save_vocab = '../../results/tokenization/vocabulary.txt'

file_to_save_corpus = '../../results/tokenization/corpus_split.csv'
# read data to a file
with open(corpus_file, encoding="utf-8") as f:
    corpus = f.read().lower()
    print("----------------------------------CORPUS----")
data_cleaner = DataCleaner(corpus)
all_words, all_sentences_split = data_cleaner.clean_content()
print ('------------------vocabulary------------------------')
print (len(all_words))
# print (all_sentences_split)
words_to_save = []
file = open(file_to_save_vocab,'w', encoding="utf8")
for word in all_words:
    file.write(word + '\n')
print (len(all_words))
print ('DONE')
