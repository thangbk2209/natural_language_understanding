#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pyvi import ViTokenizer, ViPosTagger
import re
import numpy as np
import tensorflow as tf
from data_cleaner import DataCleaner
import pandas as pd

""" 
this class demonstrate to tokenize corpus
initial function:
    window_size: number of neighbor word will become output of current word
    to train word2vec model
    corpus: corpus to tokenise
"""
class Tokenizer:
    def __init__(self, corpus = ""):
        self.corpus = corpus
    def tokenize(self):
        data_cleaner = DataCleaner(self.corpus)
        all_word, all_sentence_split = data_cleaner.clean_content()
        print ('all_word')
        print (all_word)
        # print ('all_sentence_split')
        # print (all_sentence_split)
        return all_word, all_sentence_split