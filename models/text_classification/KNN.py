import numpy as np
# from word2vec import Word2vec
import tensorflow as tf
from preprocessing_classifier import PreprocessingDataClassifier
import pickle
from sklearn.neighbors import KNeighborsClassifier
"""
this class training word2vec and receive a vector embedding and 
use this vector to train text classification by africial neural network model
"""
class Classifier:
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
    def __init__(self, n_neighbors = None,vectors = None, embedding_dim = None, input_size = None,
    word2int = None, int2word = None, file_to_save_classified_model = ""): 
        self.n_neighbors = n_neighbors
        self.file_to_save_classified_model = file_to_save_classified_model
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.vectors = vectors
        self.word2int = word2int
        self.int2word = int2word
    def classify(self, file_data_classifier):
        # Preprocessing data
        print ('------------------start preprocessing data ------------------')
        # print (self.vectors)
        preprocessing_data = PreprocessingDataClassifier(self.vectors, self.embedding_dim, self.input_size, self.word2int, self.int2word, file_data_classifier)
        # preprocessing_data = PreprocessingDataClassifier(self.vectors, self.embedding_dim, self.input_size,file_data_classifier)
        print ('----------------------start training -----------------------')
        self.x_train, self.y_train, self.x_test, self.y_test, self.test_label, self.all_sentences, self.texts = preprocessing_data.preprocessing_data_KNN()
        # print (self.x_train[0])
        # print (self.y_train[1])
        knn = KNeighborsClassifier(n_neighbors = self.n_neighbors)
        knn.fit(self.x_train, self.y_train)
        pred = knn.predict(self.x_test)
        # print (pred)
        correct = 0
        for i in range(len(pred)):
            if(pred[i] == self.y_test[i]):
                correct += 1
        print (correct/len(pred))
        pickle.dump(knn, open(self.file_to_save_classified_model, 'wb'))
        return pred
    def save_trained_classifier_data(self,data):
        with open(self.file_to_save_classified_data,'wb+') as out:
            pickle.dump(data,out,pickle.HIGHEST_PROTOCOL)
    def train(self, file_data_classifier):
        print ('-------------------file_data_classifier----------------')
        # print (file_data_classifier)
        prediction = self.classify(file_data_classifier)
        predict = []
        for i in range(len(prediction)):
            predict.append(prediction[i])
        correct = 0
        fail_file = open('../../results/text_classification/fail.txt','w',encoding="utf8")
        with open('../../data/train/train.txt') as input:
            line = input.readline()
            line = line.strip()
            temp = line.split(" ")
            train_index = [int(i) for i in temp]
            y = []
        for i in range(1445):
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
        return accuracy