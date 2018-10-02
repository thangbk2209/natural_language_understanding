from preprocessing_data import Preprocess
import tensorflow as tf
import matplotlib.pyplot as plt
import time
class BiLSTMTokenizer():
    def __init__(self, num_units = None, input_size = None, embedding_dim = None, 
                batch_size = None, epochs = None, learning_rate = None, patience = None, data_file = "", file_to_save_model = ""):
        self.num_units = num_units
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.data_file = data_file
        self.file_to_save_model = file_to_save_model
        
    def data_preprocessing(self):
        preprocess = Preprocess(self.input_size, self.data_file)
        self.number_words, self.x_train, self.y_train = preprocess.preprocessing_data()
    def init_RNN(self):
        num_layers = len(self.num_units) 
        hidden_layers = []
        for i in range(num_layers):
            cell = tf.contrib.rnn.LSTMCell(self.num_units[i], state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, 
                                 dtype=tf.float32)
            hidden_layers.append(cell)
        rnn_cells = tf.contrib.rnn.MultiRNNCell(hidden_layers, state_is_tuple = True)
        return rnn_cells
    def early_stopping(self, array, patience):
        value = array[len(array) - patience - 1]
        arr = array[len(array)-patience:]
        check = 0
        for val in arr:
            if(val > value):
                check += 1
        if(check == patience):
            return False
        else:
            return True
    def fit(self):
        print ('============start preprocessing==============')
        # print (self.data_file)
        self.data_preprocessing()
        print (self.number_words)
        # print (self.x_train)
        # print (type(self.x_train))
        print ('==================shape==================')
        print (self.x_train.shape)
        print (self.y_train.shape)
        # print (self.y_train.shape[2])
        # print (self.y_train)
        # lol
        print ('==============finish preprocessing==============')
        sentence_one_hot = tf.placeholder(tf.float32, name = 'sentence_one_hot', shape=(None, self.input_size, self.number_words))
        # out_weights=tf.Variable(tf.random_normal([self.number_words, self.embedding_dim]))
        # out_bias=tf.Variable(tf.random_normal([self.embedding_dim]))
        # prediction=tf.nn.sigmoid(tf.matmul(sentence_one_hot,out_weights)+out_bias,name="prediction")
        embedding = tf.layers.dense(sentence_one_hot, self.embedding_dim, name = "embedding")
        # embedding1 = tf.convert_to_tensor(embedding, dtype=tf.float32)
        with tf.variable_scope('LSTM_fw_layer'):
            LSTM_fw_layer = self.init_RNN()
        with tf.variable_scope('LSTM_bw_layer'):
            LSTM_bw_layer = self.init_RNN()
        output_bidirection, state = tf.nn.bidirectional_dynamic_rnn(LSTM_fw_layer, LSTM_bw_layer, embedding, dtype = 'float32' )    
        input_softmax = tf.concat([output_bidirection[0],output_bidirection[1]],2)
        # concaternate = input_softmax
        outputs = tf.layers.dense(input_softmax,self.y_train.shape[2],activation = tf.nn.softmax, name="outputs")
        y_label = tf.placeholder(tf.float32, name="y_label", shape=(None, self.y_train.shape[1] , self.y_train.shape[2]))
        # define the loss function:
        test_sum = -tf.reduce_sum(y_label * tf.log(outputs), reduction_indices=[2])
        cross_entropy_loss = tf.reduce_mean(test_sum)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy_loss,name='training_step')
        sess = tf.Session()
        init = tf.global_variables_initializer()
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths = True)
        sess.run(init)
        total_batch = int(len(self.x_train)/ self.batch_size)
        print (total_batch)
        loss_set = []
        for _ in range(self.epochs):
            start_time = time.time()
            print ('epoch: ', _ + 1)
            avg_loss = 0
            for j in range(total_batch):
                batch_x_train, batch_y_train = self.x_train[j*self.batch_size:(j+1)*self.batch_size], self.y_train[j*self.batch_size:(j+1)*self.batch_size]
                sess.run(optimizer, feed_dict={sentence_one_hot: batch_x_train, y_label: batch_y_train})
                loss = sess.run(cross_entropy_loss, feed_dict={sentence_one_hot: batch_x_train, y_label: batch_y_train})/total_batch
                # test_sum = sess.run(test_sum, feed_dict={sentence_one_hot: batch_x_train, y_label: batch_y_train})
                # print (test_sum)
                # print (loss)
                avg_loss += loss
            loss_set.append(avg_loss)
            # if ( _ > self.patience):
            #     if (self.early_stopping(loss_set, self.patience) == False):
            #         print ("early stopping tokenizer training")
            #         break
            print ("Epoch training tokenizer finished")
            
            print('loss is : ',avg_loss)
            print("finished training tokenizer phrase!!!")
            print ('time for epoch: ', _ + 1 , time.time()-start_time)
        save_path = saver.save(sess, self.file_to_save_model)

        plt.plot(loss_set)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('test.png')
        # plt.plot(cost_valid_inference_set)