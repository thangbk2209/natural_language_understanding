import numpy as np
# from word2vec import Word2vec
import tensorflow as tf
from preprocessing_LSTM_classifier import PreprocessingDataClassifier
import pickle
import matplotlib.pyplot as plt
"""
this class training word2vec and receive a vector embedding and 
use this vector to train text classification by africial neural network model
"""
class Classifier:
    CLASSIFY_BY_SOFTMAX = 1
    CLASSIFY_BY_SVM = 2
    CLASSIFY_BY_KNN = 3
    CLASSIFY_BY_MOMENTUM = 4
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
     epoch_classifier = None ,embedding_dim = None, batch_size_classifier = None, 
     optimizer_method = None, num_units_LSTM = None, file_to_save_classified_data="",history_file = ""): 
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
        self.num_units_LSTM = num_units_LSTM
        self.history_file = history_file
        # print(self.file_to_save_classified_data)
    def init_RNN(self, num_units):
        num_layers = len(num_units) 
        hidden_layers = []
        for i in range(num_layers):
            cell = tf.contrib.rnn.LSTMCell(num_units[i], state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, 
                                 dtype=tf.float32)
            hidden_layers.append(cell)
        rnn_cells = tf.contrib.rnn.MultiRNNCell(hidden_layers, state_is_tuple = True)
        return rnn_cells
    def classify(self, file_data_classifier):
        # Preprocessing data
        print ('------------------start preprocessing data ------------------')
        preprocessing_data = PreprocessingDataClassifier(self.vectors, self.embedding_dim, self.input_size, self.word2int, self.int2word, file_data_classifier)
        # preprocessing_data = PreprocessingDataClassifier(self.vectors, self.embedding_dim, self.input_size,file_data_classifier)
        print ('----------------------start training -----------------------')
        self.x_train, self.y_train, self.x_test, self.y_test, self.int2intent, self.test_label, self.all_sentences = preprocessing_data.preprocessing_data()
        print ('----------------test preprocessing LSTM---------------')
        # print (self.x_train[0])
        # Create graph
        tf.reset_default_graph()
        hidden_layer = self.init_RNN(self.num_units_LSTM)
        x = tf.placeholder(tf.float32, name="x", shape=(None, self.input_size, self.embedding_dim))
        # input_classifier = tf.reshape(x,[tf.shape(x)[0], self.input_size * self.embedding_dim])
        # hidden_value1 = tf.layers.dense(input_classifier, 256, activation = tf.nn.relu, name="hidden1")
        # hidden_value2 = tf.layers.dense(hidden_value1, 64, activation = tf.nn.relu)
        outputs,new_state = tf.nn.dynamic_rnn(hidden_layer, x,dtype="float32")
        out_weights=tf.Variable(tf.random_normal([self.input_size, self.num_classes]))
        out_bias=tf.Variable(tf.random_normal([self.num_classes]))
        prediction=tf.nn.softmax(tf.matmul(outputs[:,:,-1],out_weights)+out_bias,name="prediction")
        # # prediction = tf.layers.dense(outputs, self.num_classes, activation = tf.nn.softmax, name="prediction")
        y_label = tf.placeholder(tf.float32, name="y_label", shape=(None, self.num_classes))
        # define the loss function:
        cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
        
        #select optimizer method 
        if self.optimizer_method == self.OPTIMIZER_BY_GRADIENT:
            optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_loss,name='training_step')
        elif self.optimizer_method == self.CLASSIFY_BY_MOMENTUM:
            optimizer = tf.train.MomentumOptimizer(learning_rate = 0.01, momentum = 0.9).minimize(cross_entropy_loss,name='training_step')
        elif self.optimizer_method == self.OPTIMIZER_BY_ADAM:
            optimizer = tf.train.AdamOptimizer(0.01).minimize(cross_entropy_loss,name='training_step')
        sess = tf.Session()
        init = tf.global_variables_initializer()
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        sess.run(init) #make sure you do this!
        # train for n_iter iterations
        total_batch = int(len(self.x_train)/ self.batch_size_classifier)
        loss_set = []
        for _ in range(self.epoch_classifier):
            avg_loss = 0
            for j in range(total_batch):
                batch_x_train, batch_y_train = self.x_train[j*self.batch_size_classifier:(j+1)*self.batch_size_classifier], self.y_train[j*self.batch_size_classifier:(j+1)*self.batch_size_classifier]
                sess.run(optimizer, feed_dict={x: batch_x_train, y_label: batch_y_train})
                loss = sess.run(cross_entropy_loss, feed_dict={x: batch_x_train, y_label: batch_y_train})/total_batch
                avg_loss += loss
            loss_set.append(avg_loss)
            plt.plot(loss_set)
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.savefig(self.history_file)
            # plt.plot(cost_valid_inference_set)
            print ('epoch: ', _ + 1)
            print('loss is : ',avg_loss)
            print("finished training classification phrase!!!")
        prediction = sess.run(prediction, feed_dict={x: self.x_test})
        save_path = saver.save(sess, self.file_to_save_classified_data)
        return prediction
    def save_trained_classifier_data(self,data):
        with open(self.file_to_save_classified_data,'wb+') as out:
            pickle.dump(data,out,pickle.HIGHEST_PROTOCOL)
    def train(self, file_data_classifier):
        print ('-------------------file_data_classifier----------------')
        # print (file_data_classifier)
        prediction = self.classify(file_data_classifier)
        print (prediction)
        predict = []
        for i in range(len(prediction)):
            predict.append(self.int2intent[np.argmax(prediction[i])])
        correct = 0
        fail_file = open('../../results/text_classification/fail.txt','w',encoding="utf8")
        with open('../../data/train/train.txt') as input:
            line = input.readline()
            line = line.strip()
            temp = line.split(" ")
            train_index = [int(i) for i in temp]
            y = []
        for i in range(1430):
            # print (i)
            if i not in train_index:
                y.append(i)
        for i in range(len(predict)):
            
            if(predict[i] == self.test_label[i]):
                correct +=1
            print (y[i]+1,',',self.all_sentences[y[i]],',',self.test_label[i],',',predict[i])
            fail_file.write(str(y[i]+1)+" ")
        accuracy = correct/len(self.test_label)
        print ("accuracy: ", accuracy)
        return accuracy, self.int2intent