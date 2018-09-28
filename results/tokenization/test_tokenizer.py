import tensorflow as tf 
import pickle as pk 
import numpy as np 
from nltk.tokenize import sent_tokenize, word_tokenize
from pyvi import ViPosTagger,ViTokenizer 
import re
def read_trained_data(file_trained_data):
    with open(file_trained_data,'rb') as input_file :
        word2int = pk.load(input_file)
        int2word = pk.load(input_file)
    return word2int, int2word
def to_one_hot(data_point_index,vocab_size):
    temp = np.zeros(vocab_size, dtype = np.int8)
    temp[data_point_index] = 1
    return temp
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)
def compound_word(words):
    # this function transform output of bidirectional lstm tokenizer to format of pyvi
    tokens = []
    for i,word in enumerate(words):
        token = ''
        if(word[1] == 'B_W'):
            token += word[0]
            for j in range(i+1,len(words),1):
                if(words[j][1] == 'I_W' ):
                    token = token + '_' + words[j][0]
                else:
                    break
            tokens.append(token)
        elif(word[1] == 'O'):
            tokens.append(word[0])
        else:
            continue
    return tokens
def separate_word(tokens):
    # this function transform output of pyvi tokenizer to format of bidirectional lstm
    word_separate = []
    for token in tokens:
        words = token.split('_')
        if len(words) == 1:
            word_separate.append([words[0],'B_W'])
        else:
            arr = []
            word_separate.append([words[0],'B_W'])
            for i in range(1,len(words),1):
                word_separate.append([words[i],'I_W'])
    return word_separate

word2int, int2word = read_trained_data('word2int_ver2.pkl')
corpus_file = '../../data/corpus.txt'
with open(corpus_file, encoding = 'utf-8') as f:
    text = f.read().lower()
sentences = sent_tokenize(text)
print ("=========number of sentence============")
print (len(sentences))

sentence = ["xả ra mã ssi"]
pyvi_word = ViPosTagger.postagging(ViTokenizer.tokenize(sentence[0]))
print (pyvi_word[0])
label_word = separate_word(pyvi_word[0])
print (label_word)
not_have = []
for sentence in sentences:
    # sentence = sentence[:-1]
    print(sentence)
    # print(sentence[-1])
    
    all_single_word = word_tokenize(sentence)
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
    # p = re.compile("^\d+?\.\d+?$")
    # print (p.match("alo"))
    # lol
    number_words = len(word2int)
    input_size = 64
    x_one_hot_vector = []
    number_replace = '1000'
    for i in range(len(all_single_word)):
        # if(all_single_word[i].isdigit()):
        #     all_single_word[i] = number_replace
        # elif re.match("^\d+?\.\d+?$", all_single_word[i]) is not None:
        #     all_single_word[i] = number_replace
        if(hasNumbers(all_single_word[i])):
            all_single_word[i] = number_replace
    for i in range(len(all_single_word)):
        # print (all_single_word[i], word2int[all_single_word[i]])
        if all_single_word[i] in word2int:
            x_one_hot_vector.append(to_one_hot(word2int[all_single_word[i]], number_words)) 
        # elif len(all_single_word[i])<= 7 and hasNumbers(all_single_word[i]) == False:
        else:
            # print (all_single_word[i])
            not_have.append(all_single_word[i])
not_have = set(not_have)
print (not_have)
print (len(not_have)) 
    
    # for i in range(len(all_single_word), input_size, 1):
    #     temp = np.zeros(number_words,dtype = np.int8)
    #     x_one_hot_vector.append(temp)
    # x_data = [x_one_hot_vector]
    # x_data = np.asarray(x_data)

    # num_units = [32,8]
    # embedding_dim = 50
    # epochs = 500
    # batch_size = 128
    # learning_rate = 0.2
    # file_to_save_model = 'model_saved_ver3/model' + str(input_size) + '-' + str(num_units) + '-' + str(embedding_dim) + '-' + str(batch_size)+'.meta'

    # with tf.Session() as sess:
        
    #     #First let's load meta graph and restore weights
    #     saver = tf.train.import_meta_graph(file_to_save_model)
    #     saver.restore(sess,tf.train.latest_checkpoint('model_saved_ver3/'))
    #     # Access and create placeholders variables and
    #     graph = tf.get_default_graph()
        
    #     # print ([n.name for n in tf.get_default_graph().as_graph_def().node])
    #     x = graph.get_tensor_by_name("sentence_one_hot:0")
    #     y_label = graph.get_tensor_by_name("y_label:0")
    #     # print ([n.name for n in tf.get_default_graph().as_graph_def().node])
    #     # lol
    #     # Access the op that you want to run. 
    #     prediction = graph.get_tensor_by_name("outputs/Softmax:0")
    #     # for i in range(epochs):
    #         # for j in range(total_batch):
    #             # batch_x_train, batch_y_train = x_train[j*batch_size_classifier:(j+1)*batch_size_classifier], y_train[j*batch_size_classifier:(j+1)*batch_size_classifier]
    #     pred = (sess.run(prediction,{x:x_data}))
        
    #     index = tf.argmax(pred, axis=1, name=None)
    #     index = sess.run(index)
        
    #     corr_pred = []
    #     for i in range(len(x_one_hot_vector)):
    #         corr_pred.append(index[i])
    #     labels = []
    #     for j in range(len(all_single_word)):
    #         if (corr_pred[j] == 0):
    #             labels.append([all_single_word[j],'B_W'])
    #         elif (corr_pred[j] == 1):
    #             labels.append([all_single_word[j],'I_W'])
    #         else:
    #             labels.append([all_single_word[j],'O'])
        # print (labels)
        # tokens = compound_word(labels)
        # print (tokens)