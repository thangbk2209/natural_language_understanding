from tokenizer import Tokenizer
from data_cleaner import DataCleaner
import pandas as pd 
import numpy as np
import pickle as pk

corpus_file = '../../data/corpus_test.txt'
file_to_save_vocab = '../../results/tokenization/vocabulary_test.txt'

file_to_save_corpus = '../../results/tokenization/corpus_split.csv'
# read data to a file
with open(corpus_file, encoding="utf-8") as f:
    #for line in f:
        # print (line)
        # #temp = line.split(" ")
        # print (temp)
        # for word in temp:
        #    print("word:",word,"word encode",word.encode("utf-8"))
        # #    print(word)
        # print("---------------------------------")
    corpus = f.read().lower()
    print("----------------------------------CORPUS----")
    # print(corpus)

# window_size = 1
# tokenizer = Tokenizer(corpus)
# tokenizer.tokenize()
data_cleaner = DataCleaner(corpus)
all_words, all_sentences_split = data_cleaner.clean_content()
print ('------------------vocabulary------------------------')
print (len(all_words))
# print (all_sentences_split)
words_to_save = []
file = open(file_to_save_vocab,'w', encoding="utf8")
for word in all_words:
    # print (word)
    file.write(word + '\n')
    # print (word)
    # wordi = []
    # wordi.append(word)
    # # print (wordi)
    # words_to_save.append(wordi)
# print (words_to_save)
# wordDf = pd.DataFrame(np.array(words_to_save))
# wordDf.to_csv(file_to_save_vocab, index=False, header=None)

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
print (len(all_words))
print ('DONE')
# save_results(all_words)
