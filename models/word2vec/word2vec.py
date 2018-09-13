from preprocessing_word2vec import Preprocess_W2v
from pyvi import ViPosTagger, ViTokenizer
import tensorflow as tf
import pandas as pd 
import numpy as np
import pickle as pk
import math
"""
This class have function for training word2vec model
initial_function:
    epoch_word2vec: number of times training phase run over all training data
    embedding_dim: size of word embedding vector
    batch_size_word2vec: number of sample each update weight
    file_to_save_trained_data: file to save weight model
"""
class Word2Vec:
    def __init__(self, window_size = None, epoch_word2vec = None, embedding_dim = None, batch_size_word2vec = None, file_to_save_trained_data=""):
        self.window_size = window_size
        self.epoch_word2vec = epoch_word2vec
        self.embedding_dim = embedding_dim
        self.batch_size_word2vec = batch_size_word2vec
        self.file_to_save_trained_data = file_to_save_trained_data
    def preprocessing_data(self):
        preprocess_w2v = Preprocess_W2v(self.window_size)
        self.x_train, self.y_train, self.vocab_size, self.word2int, self.int2word = preprocess_w2v.prepare()
    def train(self):
        print ('-------------------preprocessing data -----------------------')
        self.preprocessing_data()
        print (self.x_train)
        print ('-------------------- start training word2vec --------------------')
        # print ('---------------check-----------------')
        # print (self.x_train)
        # making placeholders for x_train and y_train
        x = tf.placeholder(tf.float32, shape=(None, self.vocab_size))
        y_label = tf.placeholder(tf.float32, shape=(None, self.vocab_size))

        # EMBEDDING_DIM = 16 # you can choose your own number
        # W1 = tf.Variable(tf.random_normal([self.vocab_size, self.embedding_dim]))
        # b1 = tf.Variable(tf.random_normal([self.embedding_dim])) #bias
        # hidden_representation = tf.add(tf.matmul(x,W1), b1)
        hidden_representation = tf.layers.dense(x, self.embedding_dim)
        # W2 = tf.Variable(tf.random_normal([self.embedding_dim, self.vocab_size]))
        # b2 = tf.Variable(tf.random_normal([self.vocab_size]))
        # prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))
        pred = tf.layers.dense(hidden_representation, self.vocab_size, activation=tf.nn.softmax)
        prediction = tf.layers.dropout(
                            pred,
                            rate=0.995,
                            noise_shape=None,
                            seed=None,
                            training=False,
                            name=None
                        )
        # define the loss function:
        cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
        # define the training step:
        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init) #make sure you do this!
        total_batch = int(len(self.x_train)/self.batch_size_word2vec)
        # train for n_iter iterations
        for _ in range(self.epoch_word2vec):
            avg_loss = 0
            for j in range(total_batch):
                batch_x_train, batch_y_train = self.x_train[j*self.batch_size_word2vec: (j+1)*self.batch_size_word2vec], self.y_train[j*self.batch_size_word2vec: (j+1)*self.batch_size_word2vec]
                # print ('batch_x_train')
                # print (batch_x_train)
                # print ('batch_y_train')
                # print (batch_y_train)
                sess.run(optimizer, feed_dict={x: batch_x_train, y_label: batch_y_train})
                # prediction = sess.run([prediction], feed_dict={x: batch_x_train})
                # print ('prediction')
                # print (prediction)
                loss = sess.run(cross_entropy_loss, feed_dict={x: batch_x_train, y_label: batch_y_train})/total_batch
                # print ("loss: ", loss)
                # print ('prediction: ', prediction)
                avg_loss += loss
            print('epoch_word2vec : ', _+1)
            total_epoch = _ + 1
            print("loss: ", avg_loss)
            if math.isnan(avg_loss):
                break
        print("finished training word2vec phrase!!!")
        vocab = []
        for i in range(self.vocab_size):
            temp = np.zeros(self.vocab_size)
            temp[i] = 1
            vocab.append(temp)
        self.vectors = sess.run(hidden_representation,feed_dict={x: vocab})
        if(total_epoch == self.epoch_word2vec):
            self.save_trained_data()
        # print (self.word2int['hụt_hơi'])
        return self.vectors, self.word2int, self.int2word
    # save data to file
    def save_trained_data(self):
        with open(self.file_to_save_trained_data,'wb') as output:
            pk.dump(self.vectors,output,pk.HIGHEST_PROTOCOL)
            pk.dump(self.word2int,output,pk.HIGHEST_PROTOCOL)
            pk.dump(self.int2word,output,pk.HIGHEST_PROTOCOL)
    
    # read data to a file
    def read_trained_data(self,file_trained_data):
        try :
            with open(file_trained_data,'rb') as input_file :
                vectors = pk.load(input_file)
                word2int = pk.load(input_file)
                int2word = pk.load(input_file)
            return vectors, word2int, int2word
        except IOError:
            print ("Error: File trained data does not exist")
            print("Change to train mode (y/n):")
            char = input()
            if(char == 'y'):
                print("changing to train mode confirmed")
                print("restart in 1 seconds")
                #countdown(50)
                self.experiment.restart()
                sys.exit(1)
            else :
                print("Exit program")
                sys.exit(1)   
    def cosine_distance(self, vec1, vec2):
        a = np.subtract(vec1,vec2)
        return np.sqrt(np.sum(np.square(a)))
    def distance(self, sentence1, sentence2):
        vectors, self.word2int, self.int2word = self.read_trained_data(self.file_to_save_trained_data)
        stop_words = StopWord()
        
        vec1,vec2 = [],[]
        vec1 = stop_words.split_sentence(sentence1, vectors, self.word2int)
        vec2 = stop_words.split_sentence(sentence2, vectors, self.word2int)
        if len(vec1) < len(vec2):
            for i in range(len(vec2) - len(vec1)):
                padding = np.zeros(self.embedding_dim)
                vec1.append(padding)
        elif len(vec2) < len(vec1):
            for i in range(len(vec1) - len(vec2)):
                padding = np.zeros(self.embedding_dim)
                vec2.append(padding)
        return self.cosine_distance(vec1,vec2)
