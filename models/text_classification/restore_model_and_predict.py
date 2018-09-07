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

input_size = 16
window_size = 2
embedding_dim = 50

batch_size_word2vec = 8
file_to_save_word2vec_data = '../../results/word2vec/ver6/ws-' + str(window_size) + '-embed-' + str(embedding_dim) + 'batch_size-' + str(batch_size_word2vec) + '.pkl'

                
vectors, word2int, int2word = read_trained_data(file_to_save_word2vec_data)

texts = []
intents_data = []
intents_official = ['end', 'trade', 'cash_balance', 'advice', 'order_status', 'stock_balance', 'market', 'cancel']
sentences = {}
with open('../../results/text_classification/fail.txt', encoding="utf8") as input:
    for line in input :
        # print (line)
        temp = line.split(",",1)
        temp[1] = temp[1].lower()
        texts.append(temp[1])  #list of train_word
        intents_data.append(temp[0]) #list of label
        sentences[temp[1]] = temp[0]
intents_filter = intents_official
intents = list(intents_data)
intents_size = len(intents_filter)

def to_one_hot(index_of_intent,intent_size):
    temp = np.zeros(intent_size)
    temp[index_of_intent] = 1
    return list(temp)
intent2int = {}
int2intent = {}
x_train = []
y_train = []
all_sentences = []
for index,intent in enumerate(intents_official):
    intent2int[intent] = index
    int2intent[index] = intent 
for i, sentence in enumerate(texts):
    # print (i)
    data_cleaner = DataCleaner(sentence)
    all_words = data_cleaner.separate_sentence()
    data_x_raw = []
    # print (i)
    # print (all_words)
    for word in all_words:
        # print (word)
        data_x_raw.append(vectors[word2int[word]])
    for k in range(input_size - len(data_x_raw)):
        padding = np.zeros(embedding_dim)
        data_x_raw.append(padding)
    data_x_original = data_x_raw
    label = to_one_hot(intent2int[intents[i]], intents_size)
    x_train.append(data_x_original)
    y_train.append(label)
    all_sentences.append(all_words)

# prepare data for test
# print (data_x)
batch_size_classifier = 1
epochs = 20
total_batch = int(len(x_train)/ batch_size_classifier)

# train_x = []
# train_y = []
# create session, restore model and prediction
with tf.Session() as sess:
    
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('../../results/text_classification/ANN_ver6/ws-2-embed-50batch_size_w2c-8batch_size_cl8.meta')
    saver.restore(sess,tf.train.latest_checkpoint('../../results/text_classification/ANN_ver6/'))
    # Access and create placeholders variables and
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_label = graph.get_tensor_by_name("y_label:0")
    # Access the op that you want to run. 
    prediction = graph.get_tensor_by_name("prediction/Softmax:0")
    for i in range(epochs):
        for j in range(total_batch):
            batch_x_train, batch_y_train = x_train[j*batch_size_classifier:(j+1)*batch_size_classifier], y_train[j*batch_size_classifier:(j+1)*batch_size_classifier]
            # pred = (sess.run(prediction,{x:data_x}))
            # corr_pred = tf.reduce_max(pred)
            # index = tf.argmax(pred, axis=1, name=None)
            print (batch_x_train[0])
            print (batch_y_train[0])
            train_op = sess.graph.get_operation_by_name('training_step')
            sess.run(train_op,{x:batch_x_train,y_label: batch_y_train})
            # print sess.run(pred)
            # print (sess.run(corr_pred))
            # print (sess.run(index))
            # print (int2intent[sess.run(index)[0]])
            print ('epoch: ',i,'Done')
    save_path = saver.save(sess, '../../results/text_classification/ANN_ver6/ws-' + str(window_size) + '-embed-' + str(embedding_dim) + 'batch_size_w2c-' + str(batch_size_word2vec) + 'batch_size_cl8')
        # a = tf.maximum(pred)
        # print (sess.run(a))