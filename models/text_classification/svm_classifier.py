import numpy as np
# from word2vec import Word2vec
import tensorflow as tf
from preprocessing_classifier import PreprocessingDataClassifier
import pickle
from sklearn import svm
"""
this class training word2vec and receive a vector embedding and 
use this vector to train text classification by africial neural network model
"""
class SVM_Classifier:
    CLASSIFY_BY_SOFTMAX = 1
    CLASSIFY_BY_SVM = 2
    CLASSIFY_BY_KNN = 3
    CLASSIFY_BY_BAYESIAN = 4
    OPTIMIZER_BY_GRADIENT = 5
    OPTIMIZER_BY_SGD = 6
    OPTIMIZER_BY_ADAM = 7
    """
    This initial function define:
        input_size: The max size of a sentence, if the sentence have less word than
                    input_size, the model will add zero vector to sentence
        window_size: number of word in the left and right of current word to 
                    train word2vec
        epoch_word2vec, epoch_classifier: Number of epoch for train word2vec and text classifier
        embedding_dim: size of vector representation for each word
        num_classes : number of class(label) for training text classification
        file_to_save_classified_data : path of file to save vector presentation
    """
    def __init__(self, vectors = None, word2int = None, int2word = None, input_size = None, num_classes = None, window_size = None, 
     epoch_classifier = None ,embedding_dim = None,
    batch_size_classifier = None, optimizer_method = None, file_to_save_classified_data=""): 
        self.vectors = vectors
        self.word2int = word2int
        self.int2word = int2word
        self.input_size = input_size
        self.window_size = window_size
        self.epoch_classifier = epoch_classifier
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.file_to_save_classified_data = file_to_save_classified_data
        self.batch_size_classifier = batch_size_classifier
        self.optimizer_method = optimizer_method
        # print(self.file_to_save_classified_data)
    def classify(self, file_data_classifier,clf):
        # Preprocessing data
        print ('------------------start preprocessing data ------------------')
        preprocessing_data = PreprocessingDataClassifier(self.vectors, self.embedding_dim, self.input_size, self.word2int, self.int2word, file_data_classifier)
        # preprocessing_data = PreprocessingDataClassifier(self.vectors, self.embedding_dim, self.input_size,file_data_classifier)
        print ('----------------------start training -----------------------')
        self.x_train, self.y_train, self.x_test, self.y_test, self.int2intent, self.test_label, self.all_sentences, self.texts = preprocessing_data.preprocessing_data()
        print("y_train--------------------",self.y_train[0])
        print("x_train--------------------",self.x_train[0])
        x_train_data = [np.reshape(self.x_train,-1) for i in range(len(self.x_train))]
        x_test_data = [np.reshape(self.x_test,-1) for i in range(len(self.x_test))]
        clf.fit(x_train_data,self.y_train)
        predicion = clf.predict(x_test_data)
              
        return prediction
    def save_trained_classifier_data(self,data):
        with open(self.file_to_save_classified_data,'wb+') as out:
            pickle.dump(data,out,pickle.HIGHEST_PROTOCOL)
   

    def train(self, file_data_classifier):
        clf = svm.SVC()
        print ('-------------------file_data_classifier----------------')
        # print (file_data_classifier)
        prediction = self.classify(file_data_classifier,clf)
        predict = []
        for i in range(len(prediction)):
            predict.append(self.int2intent[prediction[i]])
        correct = 0
        fail_file = open('../../results/text_classification/fail.txt','w',encoding="utf8")
        with open('../../data/train/train.txt') as input:
            line = input.readline()
            line = line.strip()
            temp = line.split(" ")
            train_index = [int(i) for i in temp]
            y = []
        for i in range(1519):
            # print (i)
            if i not in train_index:
                y.append(i)
        for i in range(len(predict)):
            
            if(predict[i] == self.test_label[i]):
                correct +=1
            else:
                print (y[i]+1,',',self.all_sentences[y[i]],',',self.test_label[i],',',predict[i])
                fail_file.write(self.test_label[i] + ',' + self.texts[y[i]])
        accuracy = correct/len(self.test_label)
        print ('correct: ',correct)
        print ('test_label: ',len(self.test_label))
        print ("accuracy: ", accuracy)
        return accuracy, self.int2intent
