#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pyvi import ViTokenizer, ViPosTagger
import re
import numpy as np
import tensorflow as tf
# from stop_words import StopWord
import pandas as pd

class Preprocess_W2v:
    """ this class demonstrate to tokenize corpus, tranform word to one hot vector 
    and construct the training data for word2vec phrase"""
    def __init__(self, window_size = None, corpus = ""):
        self.window_size = window_size
    def prepare(self):
        file_to_save_vocab = '../../results/tokenization/vocabulary.txt'
        corpus_file = '../../results/tokenization/corpus_cleaned.txt'
        # vocab_df = pd.read_csv(file_to_save_vocab, header=None, index_col=False, usecols=[0], engine='python')
        # print (len(vocab_df.values))
        vocab_raw = []
        print ("create vocab")
        with open(file_to_save_vocab, encoding="utf8") as vocab_file:
            lines = vocab_file.readlines()
            for line in lines:
                vocab_raw.append(line.rstrip())
        vocab_size = len(vocab_raw)
        # print (vocab_size)
        vocab = []
        for i in range(vocab_size):
            vocab.append(vocab_raw[i])
        # print (vocab)
        word2int = {}   # Word and coresponding number
        int2word = {}   # integer number and coresponding word
        # vocab_size = len(words) # gives the total number of unique words
        # print (words)
        for i,word in enumerate(vocab):
            word2int[word] = i
            int2word[i] = word
        # print (".......")
        data = []
        input_words = []
        # read data to a file
        with open(corpus_file, encoding="utf8") as f:
            lines = f.readlines()
        for line in lines:
            # print (line)
            sentence = line.rstrip().split(' ')
            if sentence == '':
                continue
            for i in range(len(sentence)):
                if(sentence[i] == '\n'):
                    continue
                for j in range(max(i - self.window_size,0), min(i+self.window_size+1, len(sentence)), 1):
                    if j==i:
                        continue
                    if(sentence[j] == '\n'):
                        continue
                    data.append([sentence[i],sentence[j]])
        # print ("----------------data------------------")
        # print (data[:20])
        # print (data[0][0])
        # print (data[0][1])
        # print (word2int[ data[i][0] ])
        # print (word2int[ data[i][1] ])
        sample_Df = pd.DataFrame(np.array(data))
        sample_Df.to_csv('../../results/word2vec/sample_word2vec.csv', index=False, header=None)

        """gather the samples have the same input"""
        def to_one_hot(data_point_index,vocab_size):
            temp = np.zeros(vocab_size)
            temp[data_point_index] = 1
            return temp
        # print (to_one_hot(word2int[ data[0][0] ], vocab_size))
        # print (to_one_hot(word2int[ data[0][1] ], vocab_size))
        x_train = [] # input word
        y_train = [] # output word
        for i in range(len(data)):
            # print (data[i][0], data[i][1] )
            x_train.append(to_one_hot(word2int[ data[i][0] ], vocab_size))
            y_train.append(to_one_hot(word2int[ data[i][1] ], vocab_size))
        # convert them to numpy arrays
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        # print ('number of samples!!!')
        # print (len(x_train))
        # # print(word2int[u'trang chính'])
        # print(x_train[0])
        # print(y_train[0])
        #   print ('x_train')
        #   print (x_train[0])
        #   print (y_train[0])
        return x_train, y_train, vocab_size, word2int, int2word