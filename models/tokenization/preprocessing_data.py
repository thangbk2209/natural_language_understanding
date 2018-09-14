import numpy as np
"""gather the samples have the same input"""
class Preprocess:
    def __init__(self,input_size = None, data = ""):
        self.input_size = input_size
        self.data = data
    def to_one_hot(self,data_point_index,vocab_size):
        temp = np.zeros(vocab_size)
        temp[data_point_index] = 1
        return temp
    def preprocessing_data(self):
        x_train_raw = []
        y_train_raw = []
        all_single_word = []
        with open (self.data,encoding = 'utf-8') as acro_file:
            lines = acro_file.readlines()
            x_traini = []
            y_traini = []
            for line in lines:
                # print (line)
                if line == '\n':
                    # print(1)
                    x_train_raw.append(x_traini)
                    y_train_raw.append(y_traini)
                    x_traini = []
                    y_traini = []
                else:
                    datai = line.rstrip('\n').split('\t')
                    # print(datai)
                    x_traini.append(datai[0])
                    y_traini.append(datai[1])
                    if (datai[0] not in all_single_word):
                        all_single_word.append(datai[0])

        word2int = {}   # Word and coresponding number
        int2word = {}   # integer number and coresponding word
        number_words = len(all_single_word) # gives the total number of unique words
        # print (words)
        for i,word in enumerate(all_single_word):
            word2int[word] = i
            int2word[i] = word
        labels = ['B_W','I_W','O']
        number_labels = len(labels)
        label2int = {}   # label and coresponding number
        int2label = {}   # integer number and coresponding label
        for i,label in enumerate(labels):
            label2int[label] = i
            int2label[i] = label
        x_train = []
        y_train = []
        for i in range(len(x_train_raw)):
            x_traini = []
            y_traini = []
            for j in range(len(x_train_raw[i])):
                x_traini.append(self.to_one_hot(word2int[x_train_raw[i][j]],number_words))
                y_traini.append(self.to_one_hot(label2int[y_train_raw[i][j]],number_labels))
            if(len(x_train_raw[i]) < self.input_size):
                for t in range(len(x_train_raw[i]),self.input_size,1):
                    temp = np.zeros(number_words)
                    label = np.array([1/3,1/3,1/3])
                    x_traini.append(temp)
                    y_traini.append(label)
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