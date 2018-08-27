from data_cleaner import DataCleaner 
from preprocessing_classifier import PreprocessingDataClassifier
import tensorflow as tf 
import pickle as pk 
import numpy as np 
def read_trained_data(file_trained_data):
    with open(file_trained_data,'rb') as input_file :
        vectors = pk.load(input_file)
        word2int = pk.load(input_file)
        int2word = pk.load(input_file)
    return vectors, word2int, int2word
input = "bỏ qua lệnh"
input_size = 16
window_size = 2
embedding_dim = 32

batch_size_word2vec = 4
file_to_save_word2vec_data = '../../results/word2vec/ver4/ws-' + str(window_size) + '-embed-' + str(embedding_dim) + 'batch_size-' + str(batch_size_word2vec) + '.pkl'

                
vectors, word2int, int2word = read_trained_data(file_to_save_word2vec_data)
# prepare data for test
data_cleaner = DataCleaner(input)
all_words, all_sentences_split = data_cleaner.clean_content()
print (all_words)
data_x_raw = []
for word in all_words:
    data_x_raw.append(vectors[word2int[word]])
for k in range(input_size - len(data_x_raw)):
    padding = np.zeros(embedding_dim)
    data_x_raw.append(padding)
data_x =[]
data_x.append(data_x_raw)
# print (data_x)
int2intent = {0: 'end', 1: 'trade', 2: 'cash_balance', 3: 'advice', 4: 'order_status', 5: 'stock_balance', 6: 'market',7: 'cancel'}
# create session, restore model and prediction
with tf.Session() as sess:
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('../../results/text_classification/ANN_ver3_2layer_test_pretrain/ws-2-embed-32batch_size_w2c-8batch_size_cl4.meta')
    saver.restore(sess,tf.train.latest_checkpoint('../../results/text_classification/ANN_ver3_2layer_test_pretrain/'))
    # Access and create placeholders variables and
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    # Access the op that you want to run. 
    prediction = graph.get_tensor_by_name("prediction/Softmax:0")
    pred = (sess.run(prediction,{x:data_x}))
    corr_pred = tf.reduce_max(pred)
    index = tf.argmax(pred, axis=1, name=None)
    # print sess.run(pred)
    print (sess.run(corr_pred))
    print (sess.run(index))
    print (int2intent[sess.run(index)[0]])
    # a = tf.maximum(pred)
    # print (sess.run(a))