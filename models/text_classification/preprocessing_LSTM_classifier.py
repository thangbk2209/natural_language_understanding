#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pyvi import ViTokenizer, ViPosTagger
import numpy as np 
import re
from data_cleaner import DataCleaner
class PreprocessingDataClassifier:
    """ This class prepare data for text_classification phrase.
    first, read the training data from text file: file_data_classifier.
    then, tokenize sentence from data and create vector for training phrase.
    using input_size to padding vector zeros for every vector => input from 
    each sample will have the same size
    """
    def __init__(self, vectors = None, embedding_dim = None, input_size = None,
    word2int = None, int2word = None, file_data_classifier =""):
        self.vectors = vectors
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.file_data_classifier = file_data_classifier
        self.word2int = word2int
        self.int2word = int2word
    def preprocessing_data(self):
        # stop_words = StopWord()
        texts = []
        intents_data = [] # danh sách intents trong bộ dữ liệu
        intents_official = ['end', 'trade', 'cash_balance', 'advice', 'order_status', 'stock_balance', 'market', 'cancel']
        sentences = {}
        with open(self.file_data_classifier, encoding="utf8") as input:
            for line in input :
                # print (line)
                temp = line.split(",",1)
                temp[1] = temp[1].lower()
                texts.append(temp[1])  #list of train_word
                intents_data.append(temp[0]) #list of label
                sentences[temp[1]] = temp[0]
        intents_filter = intents_official
        intents = list(intents_data)
        intents_size = len(intents_filter)
        # print (intents)
        """
        create vector one hot for label(intent)
        """
        def to_one_hot(index_of_intent,intent_size):
            temp = np.zeros(intent_size)
            temp[index_of_intent] = 1
            return list(temp)
        intent2int = {}
        int2intent = {}
        
        x_train = []
        y_train = []
        all_sentences = []
        for index,intent in enumerate(intents_filter):
            intent2int[intent] = index
            int2intent[index] = intent 
        for i, sentence in enumerate(texts):
            # print (i)
            data_cleaner = DataCleaner(sentence)
            all_words = data_cleaner.separate_sentence()
            data_x_raw = []
            for word in all_words:
                # print (word)
                data_x_raw.append(self.vectors[self.word2int[word]])
            for k in range(self.input_size - len(data_x_raw)):
                padding = np.zeros(self.embedding_dim)
                data_x_raw.append(padding)
            data_x_original = data_x_raw
            label = to_one_hot(intent2int[intents[i]], intents_size)

            x_train.append(data_x_original)
            y_train.append(label)
            all_sentences.append(all_words)
        data_classifier_size = len(x_train)
        train_size = int(data_classifier_size * 0.8)
        with open('../../data/train/train.txt') as input:
            line = input.readline()
            line = line.strip()
            temp = line.split(" ")
            train_index = [int(i) for i in temp]
           # print(train_index)
        test_label = []
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        # train_x = x_train
        # train_y = y_train 
        for i in train_index:
            # print (i)
            train_x.append(x_train[i])
            train_y.append(y_train[i])
            
        for i in range(data_classifier_size):
            
            if i not in train_index:
                test_label.append(intents[i])
                test_x.append(x_train[i])
                test_y.append(y_train[i])
        return train_x, train_y, test_x, test_y, int2intent,test_label, all_sentences