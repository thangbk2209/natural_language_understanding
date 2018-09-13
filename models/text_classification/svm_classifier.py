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
        #print("y_train--------------------",self.y_train[0])
       # print("x_train--------------------",self.x_train[0])
        x_train_data = [np.reshape(self.x_train[i],-1) for i in range(len(self.x_train))]
        print("x train data",x_train_data[0],"len x train",len(x_train_data[0]))
        #print("x train data",self.x_train.shape,"len x train",len(x_train_data[1]))
        x_test_data = [np.reshape(self.x_test[i],-1) for i in range(len(self.x_test))]
        y_train_data = [self.onehot_to_number(self.y_train[i]) for i in range(len(self.y_train))]

       # print("y train",self.y_train[0],"len y train",len(self.y_train[0]))
       # print("y train data",y_train_data[0],"len y train",len(y_train_data))
        clf.fit(x_train_data,y_train_data)
        prediction = clf.predict(x_test_data)
        print("sd",y_train_data)
        #spickle.dump(data,out,pickle.HIGHEST_PROTOCOL)
        return prediction
    def onehot_to_number(self,vector):
        for i in range(len(vector)) :
            if vector[i] :
                return i
        return 0
    def train(self, file_data_classifier):
        clf = svm.LinearSVC()
        print ('-------------------file_data_classifier----------------')
        # print (file_data_classifier)
        prediction = self.classify(file_data_classifier,clf)
        predict = []
        for i in range(len(prediction)):
            print("prediction[",i,"] =",prediction[i])
            print("intent[",i,"] =",self.int2intent[round(prediction[i])])
            predict.append(self.int2intent[round(prediction[i])])
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
        return accuracy, self.int2intent
