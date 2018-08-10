import numpy as np
from Word2vec import Word2vec
import tensorflow as tf
from preprocess_data_classifier import PreprocessingDataClassifier
import pickle
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
    def __init__(self, input_size = None, num_classes = None, window_size = None, epoch_word2vec = None, epoch_classifier = None ,embedding_dim = None,file_to_save_classified_data=""):
        self.input_size = input_size
        self.window_size = window_size
        self.epoch_word2vec = epoch_word2vec
        self.epoch_classifier = epoch_classifier
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.file_to_save_classified_data = file_to_save_classified_data
        print(self.file_to_save_classified_data)
    
    def classifier(self,vectors,optimizer_method = 0,file_data_classifier ="",batch_size_classifier = 0, validation_size = 0):
        # Preprocessing data
        preprocessing_data = PreprocessingDataClassifier(vectors, self.embedding_dim, self.input_size,file_data_classifier)
        self.x_train, self.y_train, self.x_test, self.y_test, self.int2intent, self.test_label = preprocessing_data.preprocessing_data(self.word2int, self.int2word)
        # Create graph
        x = tf.placeholder(tf.float32, shape=(None, self.input_size, self.embedding_dim))
        input_classifier = tf.reshape(x,[tf.shape(x)[0], self.input_size * self.embedding_dim])
        hidden_value = tf.layers.dense(input_classifier, 2 * self.input_size * self.embedding_dim, activation = tf.nn.sigmoid)
        prediction = tf.layers.dense(hidden_value, 3, activation = tf.nn.softmax)
        y_label = tf.placeholder(tf.float32, shape=(None, self.num_classes))
        # define the loss function:
        cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
        
        #select optimizer method 
        if optimizer_method == self.OPTIMIZER_BY_GRADIENT:
            optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)
        elif optimizer_method == self.OPTIMIZER_BY_SGD:
            a = 0
        elif optimizer_method == self.OPTIMIZER_BY_ADAM:
            a = 0
       
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init) #make sure you do this!
        # train for n_iter iterations
        total_batch = int(len(self.x_train)/ batch_size_classifier)
        for _ in range(self.epoch_classifier):
            avg_loss = 0
            for j in range(total_batch):
                batch_x_train, batch_y_train = self.x_train[j*batch_size_classifier:(j+1)*batch_size_classifier], self.y_train[j*batch_size_classifier:(j+1)*batch_size_classifier]
                sess.run(optimizer, feed_dict={x: batch_x_train, y_label: batch_y_train})
                loss = sess.run(cross_entropy_loss, feed_dict={x: batch_x_train, y_label: batch_y_train})/total_batch
                avg_loss += loss
            #print("epoch: ",_)
            print('loss is : ',avg_loss)
            print("finished training classification phrase!!!")
        prediction = sess.run(prediction, feed_dict={x: self.x_test})
        # self.save_trained_classifier_data(sess.run(W))
        # self.save_trained_classifier_data(sess.run(b))
        return prediction
    # def save_trained_classifier_data(self,data):
    #     with open(self.file_to_save_classified_data,'wb+') as out:
    #         pickle.dump(data,out,pickle.HIGHEST_PROTOCOL)

    def train(self, file_data_classifier, file_data_word2vect, optimizer_method, batch_size_classifier, batch_size_word2vec,train,file_to_save_trained_data,experiment):
        
        word2vec = Word2vec( self.window_size, self.epoch_word2vec, self.embedding_dim, file_data_word2vect, batch_size_word2vec,file_to_save_trained_data,experiment)
        if train == 1:
            vectors, self.word2int, self.int2word = word2vec.train()
        elif train == 0 :
            file_trained_data = file_to_save_trained_data
            vectors, self.word2int, self.int2word = word2vec.read_trained_data(file_trained_data)
        # sentence1 = "đi chơi với mình nhé"
        # sentence2 = "nền kinh tế hà nội"
        # print (sentence2)
        # print (word2vec.distance(sentence1, sentence2))
        prediction = self.classifier(vectors, optimizer_method, file_data_classifier, batch_size_classifier)
        # print ('vectors')
        # print (vectors)
        # print (self.word2int)
        # print (type(prediction))
        # print (prediction)
        predict = []
        for i in range(len(prediction)):
            predict.append(self.int2intent[np.argmax(prediction[i])])
        #print ("predict: ",predict,"leng predict",len(predict))
        #print (self.test_label[0] == predict[0])
        correct = 0
        for i in range(len(predict)):
            if(predict[i] == self.test_label[i]):
                correct +=1
       # print (" test index: ",self.test_label)
     #   print ("len test: ",len(self.test_label))
        accuracy = correct/len(self.test_label)
        print ("accuracy: ", accuracy)
        return accuracy