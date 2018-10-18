import numpy as np
import pickle as pk
import pandas as pd
import re
"""gather the samples have the same input"""
class Preprocess:
    def __init__(self,input_size = None, data = ""):
        self.input_size = input_size
        self.data = data
    def to_one_hot(self,data_point_index,vocab_size):
        # this function create vector onehot for word
        temp = np.zeros(vocab_size, dtype = np.int8)
        temp[data_point_index] = 1
        return temp
    def hasNumbers(self,inputString):
        # this function check inputString contain number or not. 
        # If contain number, return true
        # else return fasle
        return any(char.isdigit() for char in inputString)
    def check_Number(self,inputString):
        # this function check inputString is number or not
        if(inputString.isdigit or re.match("^\d+?\.\d+?$", inputString) is not None):
            return True
        
    # def execute_exception(self,array):
    #     # this function execute strings that have number
    #     number_replace = '1000'
    #     outputs = []
    #     for word in array:
    #         if (self.hasNumbers(word)):
    #             outputs.append(number_replace)
    #         else:
    #             outputs.append(word)
    #     return outputs
    def preprocessing_data(self):
        x_train_raw = []
        y_train_raw = []
        all_single_word = []
        number_digit = 0
        with open (self.data,encoding = 'utf-8') as acro_file:
            lines = acro_file.readlines()
            x_traini = []
            y_traini = []
            number_replace = '1000'
            for line in lines:
                # print (line)
                if line == '\n' or line == '\t\n':
                    # print(1)
                    if x_traini not in x_train_raw:
                        x_train_raw.append(x_traini)
                        y_train_raw.append(y_traini)
                    # else:
                        # print (x_traini)
                    x_traini = []
                    y_traini = []
                else:
                    datai = line.rstrip('\n').split('\t')
                    # print(datai)
                    
                    y_traini.append(datai[1])
                    if self.hasNumbers(datai[0]):
                        x_traini.append(number_replace)
                        number_digit +=1
                        if(number_digit == 1):
                            all_single_word.append(number_replace)
                        
                    else:
                        x_traini.append(datai[0])
                        if (datai[0] not in all_single_word):
                            all_single_word.append(datai[0])
        print (len(x_train_raw))
        # lol
        all_single_word_df = pd.DataFrame(all_single_word)
        all_single_word_df.to_csv("../../results/tokenization/all_single_word.csv",header=None)
        word2int = {}   # Word and coresponding number
        int2word = {}   # integer number and coresponding word
        number_words = len(all_single_word) # gives the total number of unique words
        print (number_words)
        # lol76
        for i,word in enumerate(all_single_word):
            word2int[word] = np.int16(i)
            int2word[np.int16(i)] = word
        print (int2word[0])
        print (int2word[10])
        with open('../../results/tokenization/word2int_ver12.pkl','wb') as output:
            pk.dump(word2int,output,pk.HIGHEST_PROTOCOL)
            pk.dump(int2word,output,pk.HIGHEST_PROTOCOL)
        labels = ['B_W','I_W','O']
        number_labels = len(labels)
        label2int = {}   # label and coresponding number
        int2label = {}   # integer number and coresponding label
        for i,label in enumerate(labels):
            label2int[label] = np.int16(i)
            int2label[np.int16(i)] = label
        x_train = []
        y_train = []
        dem = 0
        x_one_hot_vector = {}
        y_one_hot_vector = {}
        for i in range(len(all_single_word)):
            x_one_hot_vector[word2int[all_single_word[i]]] = self.to_one_hot(word2int[all_single_word[i]], number_words)
        for i in range(len(labels)):
            y_one_hot_vector[label2int[labels[i]]] = self.to_one_hot(label2int[labels[i]], number_labels)
        # print (y_one_hot_vector)
        # dem = 0
        for i in range(len(x_train_raw)):
            if(len(x_train_raw[i]) <= self.input_size):
                x_traini = []
                y_traini = []
                for j in range(len(x_train_raw[i])):
                    # dem+=1
                    # if(len(x_train_raw[i]) > self.input_size):
                    #     print (len(x_train_raw[i]))
                    #     print (dem)
                    x_traini.append(x_one_hot_vector[word2int[x_train_raw[i][j]]] )
                    # print (label2int[y_train_raw[i][j]])
                    y_traini.append(y_one_hot_vector[label2int[y_train_raw[i][j]]])
                    # print (y_one_hot_vector[label2int[y_train_raw[i][j]]])
                if(len(x_train_raw[i]) < self.input_size):
                    temp = np.zeros((self.input_size - len(x_train_raw[i]),number_words),dtype = np.int8)
                    # for t in range(len(x_train_raw[i]),self.input_size,1):
                    #     label = np.array([1/3,1/3,1/3])
                    label = np.full((self.input_size - len(x_train_raw[i]),len(labels)), 1/3)
                #     # label 
                for k in range(self.input_size - len(x_train_raw[i])):
                    y_traini.append(label[k])
                    x_traini.append(temp[k])
        
            x_train.append(x_traini)
            y_train.append(y_traini)
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        # for i in range(len(x_train)):
        #     print (x_train[i].shape())
        # print (x_train)
        # print (type(x_train))
        # lol
        # print (type(x_train))
        # print (x_train)
        # lol
        # print(all_single_word)
        # print(x_train_raw)
        # print(y_train_raw)
        # print(x_train)
        # print(y_train)
        return number_words, x_train, y_train