import tensorflow as tf 
import pickle as pk 
import numpy as np 
from nltk.tokenize import sent_tokenize, word_tokenize
def read_trained_data(file_trained_data):
    with open(file_trained_data,'rb') as input_file :
        word2int = pk.load(input_file)
        int2word = pk.load(input_file)
    return word2int, int2word
def to_one_hot(data_point_index,vocab_size):
    temp = np.zeros(vocab_size, dtype = np.int8)
    temp[data_point_index] = 1
    return temp
word2int, int2word = read_trained_data('word2int_ver2.pkl')
sentence = ["tôi muốn đặt lệnh mua cổ phiếu SSI với giá 23.9 và khối lượng 500 cổ phiếu"]
all_single_word = word_tokenize(sentence[0])
# print (all_single_word)
# with open ('../../data/tokenize/test.txt',encoding = 'utf-8') as acro_file:
#     lines = acro_file.readlines()
#     x_test_raw = []
#     y_test_raw = []
#     x_testi = []
#     y_testi = []
#     for line in lines:
#         # print (line)
#         if line == '\n' or line == '\t\n':
#             # print(1)
#             x_test_raw.append(x_testi)
#             y_test_raw.append(y_testi)
#             x_testi = []
#             y_testi = []
#         else:
#             datai = line.rstrip('\n').split('\t')
#             # print(datai)          
#             y_testi.append(datai[1])
#             x_testi.append(datai[0])
# print (x_test_raw[0])
# print (y_test_raw[0])
# lol
number_words = len(word2int)
input_size = 64
x_one_hot_vector = []
number_replace = '1000'
for i in range(len(all_single_word)):
    if(all_single_word[i].isdigit()):
        all_single_word[i] = number_replace
for i in range(len(all_single_word)):
    print (all_single_word[i], word2int[all_single_word[i]])
    x_one_hot_vector.append(to_one_hot(word2int[all_single_word[i]], number_words)) 
for i in range(len(all_single_word), input_size, 1):
    temp = np.zeros(number_words,dtype = np.int8)
    x_one_hot_vector.append(temp)
x_data = [x_one_hot_vector]
x_data = np.asarray(x_data)

num_units = [32,8]
embedding_dim = 50
epochs = 500
batch_size = 256
learning_rate = 0.2
file_to_save_model = 'model_saved_ver2/model' + str(input_size) + '-' + str(num_units) + '-' + str(embedding_dim) + '-' + str(batch_size)+'.meta'

with tf.Session() as sess:
    
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(file_to_save_model)
    print (saver)
    saver.restore(sess,tf.train.latest_checkpoint('model_saved_ver2/'))
    # Access and create placeholders variables and
    graph = tf.get_default_graph()
    
    # print ([n.name for n in tf.get_default_graph().as_graph_def().node])
    x = graph.get_tensor_by_name("sentence_one_hot:0")
    y_label = graph.get_tensor_by_name("y_label:0")
    # print ([n.name for n in tf.get_default_graph().as_graph_def().node])
    # lol
    # Access the op that you want to run. 
    prediction = graph.get_tensor_by_name("outputs/Softmax:0")
    # for i in range(epochs):
        # for j in range(total_batch):
            # batch_x_train, batch_y_train = x_train[j*batch_size_classifier:(j+1)*batch_size_classifier], y_train[j*batch_size_classifier:(j+1)*batch_size_classifier]
    pred = (sess.run(prediction,{x:x_data}))
    print (pred)
    index = tf.argmax(pred, axis=1, name=None)
    index = sess.run(index)
    print (index)
    corr_pred = []
    for i in range(len(x_one_hot_vector)):
        corr_pred.append(index[i])
    labels = []
    for j in range(len(all_single_word)):
        if (corr_pred[j] == 0):
            labels.append([all_single_word[j],'B_W'])
        elif (corr_pred[j] == 1):
            labels.append([all_single_word[j],'I_W'])
        else:
            labels.append([all_single_word[j],'O'])
    print (labels)
    # corr_pred = tf.reduce_max(pred)
    # index = tf.argmax(pred, axis=1, name=None)
    # corr = sess.run(corr_pred)
    # print (corr)
    # ind = sess.run(index)
            # print (batch_x_train[0])
            # print (batch_y_train[0])
            # train_op = sess.graph.get_operation_by_name('training_step')
            # sess.run(train_op,{x:batch_x_train,y_label: batch_y_train})
            # print sess.run(pred)
            # print (sess.run(corr_pred))
    # print (sess.run(corr_pred))
    # print (sess.run(index))
    # for i in range(len(ind)):
    #     print (pred[i][ind[i]])
        # print (sess.run(corr_pred)[i])
        # print (int2intent[ind[i]])
            # print ('epoch: ',i,'Done')
    # save_path = saver.save(sess, '../../results/text_classification/ANN_ver6/ws-' + str(window_size) + '-embed-' + str(embedding_dim) + 'batch_size_w2c-' + str(batch_size_word2vec) + 'batch_size_cl8')
        # a = tf.maximum(pred)
        # print (sess.run(a))