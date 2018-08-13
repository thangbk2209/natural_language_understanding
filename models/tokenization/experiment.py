from tokenizer import Tokenizer
from data_cleaner import DataCleaner
import pandas as pd 
import numpy as np
import pickle as pk
corpus_file = '../../data/corpus_test.txt'
file_to_save_vocab = '../../results/tokenization/vocabulary.csv'
file_to_save_corpus = '../../results/tokenization/corpus_split.csv'
# read data to a file
with open(corpus_file, encoding="utf8") as f:
    corpus = f.read().lower()
# window_size = 1
# tokenizer = Tokenizer(corpus)
# tokenizer.tokenize()
data_cleaner = DataCleaner(corpus)
all_words, all_sentences_split = data_cleaner.clean_content()
print ('------------------vocabulary------------------------')
# print (all_sentences_split)
words_to_save = []
for word in all_words:
    # print (word)
    wordi = []
    wordi.append(word)
    # print (wordi)
    words_to_save.append(wordi)
# print (words_to_save)
wordDf = pd.DataFrame(np.array(words_to_save))
wordDf.to_csv(file_to_save_vocab, index=False, header=None)

# all_sentences = []
# for sentence in all_sentences_split:
#     # print (sentence)
#     sentencei = []
#     for word in sentence:
#         # print (word)
#         sentencei.append([word])
#     print (sentencei)
#     all_sentences.append(sentencei)
# print (all_sentences)
# Corpus_Df = pd.DataFrame(np.array(all_sentences))
# Corpus_Df.to_csv(file_to_save_corpus, index=False, header=None)
print ('DONE')
# save_results(all_words)
