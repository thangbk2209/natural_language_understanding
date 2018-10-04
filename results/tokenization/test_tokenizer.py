import tensorflow as tf 
import pickle as pk 
import numpy as np 
from nltk.tokenize import sent_tokenize, word_tokenize
# from pyvi import ViPosTagger,ViTokenizer 
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
    print (words)
    print (len(words))
    for i,word in enumerate(words):
        print(word)
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
    print (tokens)
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


word2int, int2word = read_trained_data('word2int_ver7.pkl')
corpus_file = '../../data/corpus.txt'

def tokenize_corpus():
    with open(corpus_file, encoding = 'utf-8') as f:
        text = f.read().lower()
    sentences = sent_tokenize(text)
    print ("=========number of sentence============")
    print (len(sentences))

    # sentence = ["xả ra mã ssi"]
    # pyvi_word = ViPosTagger.postagging(ViTokenizer.tokenize(sentence[0]))
    # print (pyvi_word[0])
    # label_word = separate_word(pyvi_word[0])
    # print (label_word)
    not_have = []
    all_tokens = []
    number_sentence = 0
    x_one_hot_vector = []
    real_word = []
    file_word = open('word.txt','w', encoding="utf8")
                
    for sentence in sentences:
        sentence = sentence[:-1]
        # print(sentence)
        # print(sentence[-1])
        number_sentence+=1
        # print (number_sentence)
        all_single_word = word_tokenize(sentence)
        file_word.write(str(all_single_word) +'\n')
        file_word.write('\n')
        if(len(all_single_word)<=64):
            real_word.append(all_single_word)
    print (len(real_word))
    real_word = np.asarray(real_word)
    number_words = len(word2int)
    input_size = 64
    
    number_replace = '1000'
    for i in range(len(real_word)):
        all_single_word = []
        x_one_hot_vectori = []
        for j in range(len(real_word[i])):
            # if(all_single_word[i].isdigit()):
            #     all_single_word[i] = number_replace
            # elif re.match("^\d+?\.\d+?$", all_single_word[i]) is not None:
            #     all_single_word[i] = number_replace
            if(hasNumbers(real_word[i][j])):
                all_single_word.append(number_replace) 
            else:
                all_single_word.append(real_word[i][j])
        for i in range(len(all_single_word)):
            # print (all_single_word[i], word2int[all_single_word[i]])
            if all_single_word[i] in word2int:
                x_one_hot_vectori.append(to_one_hot(word2int[all_single_word[i]], number_words)) 
            # elif len(all_single_word[i])<= 7 and hasNumbers(all_single_word[i]) == False:
            else:
                # print (all_single_word[i])
                not_have.append(all_single_word[i])
                x_one_hot_vectori.append(to_one_hot(word2int[number_replace], number_words))
    #             # not_have = set(not_have)
    #             # print (not_have)
    #             # print (len(not_have)) 
        for i in range(len(all_single_word), input_size, 1):
            temp = np.zeros(number_words,dtype = np.int8)
            x_one_hot_vectori.append(temp)
    #         x_one_hot_vectori = np.asarray(x_one_hot_vectori)
    #         print (x_one_hot_vectori.shape)
        x_one_hot_vector.append(x_one_hot_vectori)
    # # x_data = [x_one_hot_vector]
    x_data = np.asarray(x_one_hot_vector)
    print (x_data.shape)
    # real_word = np.asarray(real_word)
    # print (x_data.shape)
    # # lol

    num_units = [32,8]
    embedding_dim = 50
    epochs = 500
    batch_size = 128
    learning_rate = 0.2
    file_to_save_model = 'model_saved_ver5/model' + str(input_size) + '-' + str(num_units) + '-' + str(embedding_dim) + '-' + str(batch_size)+'.meta'

    with tf.Session() as sess:
        
        #First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph(file_to_save_model)
        saver.restore(sess,tf.train.latest_checkpoint('model_saved_ver5/'))
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
        print (pred.shape)
        index = tf.argmax(pred, axis=1, name=None)
        index = sess.run(index)

        print (index[0:20])
        file = open('token.txt','w', encoding="utf8")
        # k = 0
        # for i in range(len(real_word)):
        #     file.write('\n')
        #     for j in range(len(real_word[i])):
        #         file.write(real_word[i][j] + '\t' + str(index[i*64+j]) +'\n')
                # k +=1
        # print (k)
    #     print (index.shape)
        labels = []
    #     k = 0
        for i in range(real_word.shape[0]):
            labelsi = []
            for j in range(len(real_word[i])):
                if (index[i*64+j] == 0):
                    labelsi.append([real_word[i][j],'B_W'])
                elif (index[i*64+j] == 1):
                    labelsi.append([real_word[i][j],'I_W'])
                else :
                    labelsi.append([real_word[i][j],'O'])
            labels.append(labelsi)
    #     # print (labels[:5])
    #     # print (labels[:-5])
    #     print (labels)
        labels = np.asarray(labels)
        print (labels.shape)
        for label in labels:
            
            print (label)
            tokens = compound_word(label)
    #     # print (tokens)
            all_tokens.append(tokens)
            for token in tokens:
                file.write(token + '\n')
            file.write('\n')
    return all_tokens
def preprocessing_testdata():
    with open ('../../data/tokenize/10k - 5.txt',encoding = 'utf-8') as acro_file:
        lines = acro_file.readlines()
        x_test_raw = []
        y_test_raw = []
        x_testi = []
        y_testi = []
        x_reali = []
        number_replace = "1000"
        number_digit = 0
        all_single_word = []
        x_real = []
        for line in lines:
            # print (line)
           
            if line == '\n' or line == '\t\n':
                # print(1)
                x_test_raw.append(x_testi)
                y_test_raw.append(y_testi)
                x_real.append(x_reali)
                x_testi = []
                y_testi = []
                x_reali = []
                
            else:
                datai = line.rstrip('\n').split('\t')
                if(hasNumbers(datai[0]) and datai[1] == "O"):
                    print (datai)
                # print(datai)  
                x_reali.append(datai[0])        
                y_testi.append(datai[1])
                if hasNumbers(datai[0]):
                    x_testi.append(number_replace)
                    number_digit +=1
                    if(number_digit == 1):
                        all_single_word.append(number_replace)
                        
                else:
                    x_testi.append(datai[0])
                    if (datai[0] not in all_single_word):
                        all_single_word.append(datai[0])
    # p = re.compile("^\d+?\.\d+?$")
    # print (p.match("alo"))
    return all_single_word, x_test_raw, y_test_raw, x_real
def evaluate():
    all_single_word, x_test_raw, y_test_raw, x_real = preprocessing_testdata()
    # lol
    print (x_test_raw[0])
    print (y_test_raw[0])
    x_one_hot_vector = []
    x_vector = []
    x_real_vector = []
    y_vector = []
    number_replace = '1000'
    number_words = len(word2int)
    input_size = 64
    for i,x_testi in enumerate(x_test_raw):
        x_one_hot_vectori = []
        y_vectori = []
        x_real_vectori = []
        x_vectori = []
        if(len(x_test_raw[i]) > input_size):
            continue
        else:
            for j in range(len(x_test_raw[i])):
                # print (all_single_word[i], word2int[all_single_word[i]])
                if x_test_raw[i][j] in word2int:
                    x_one_hot_vectori.append(to_one_hot(word2int[x_test_raw[i][j]], number_words))
                    x_real_vectori.append(x_real[i][j])
                    x_vectori.append(x_test_raw[i][j])
                    y_vectori.append(y_test_raw[i][j])
                else:
                    x_one_hot_vectori.append(to_one_hot(word2int[number_replace], number_words))
                    x_vectori.append(number_replace)
                    x_real_vectori.append(x_real[i][j])
                    y_vectori.append(y_test_raw[i][j])
            for j in range(len(x_test_raw[i]), input_size, 1):
                temp = np.zeros(number_words,dtype = np.int8)
                x_one_hot_vectori.append(temp)
                # y_vectori.append(y_test_raw[i][j])
        x_one_hot_vector.append(x_one_hot_vectori)
        x_real_vector.append(x_real_vectori)
        x_vector.append(x_vectori)
        y_vector.append(y_vectori)
    x_data = np.asarray(x_one_hot_vector)
    y_vector = np.asarray(y_vector)
    x_vector = np.asarray(x_vector)
    x_real_vector = np.asarray(x_real_vector)
    print (x_data.shape)
    # y_test_raw = np.asarray(y_test_raw)
    num_units = [32,8]
    embedding_dim = 50
    epochs = 500
    batch_size = 128
    learning_rate = 0.2
    file_to_save_model = 'model_saved_ver7/model' + str(input_size) + '-' + str(num_units) + '-' + str(embedding_dim) + '-' + str(batch_size)+'.meta'

    with tf.Session() as sess:
        
        #First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph(file_to_save_model)
        saver.restore(sess,tf.train.latest_checkpoint('model_saved_ver7/'))
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
        print (pred.shape)
        # lol
        index = tf.argmax(pred, axis=1, name=None)
        index = sess.run(index)
        print (index.shape)
        # print (index)
        # x_test_raw = np.asarray(x_test_raw)
        # x_test_raw_final = []
        # y_test_raw_final = []
        # print (x_test_raw.shape)
        # for i in range(len(x_test_raw)):
        #     if(len(x_test_raw[i]) <= input_size):
        #         x_test_raw_final.append(x_test_raw[i])
        #         y_test_raw_final.append(y_test_raw[i])
        # x_test_raw_final = np.asarray(x_test_raw_final)
        # y_test_raw_final = np.asarray(y_test_raw_final)
        # print (x_test_raw_final.shape)
        # print (y_test_raw_final.shape)
        # print (x_test_raw_final[-1])
        # lol
        # print (y_test_raw_final[-1])
        special_character = [".","x","“","”","…",">","<","@", "#", ")","(","+","-","_", "&","=","•","©","{", "}", "±", "v.v...","," ]
        labels = []
        for i in range(x_data.shape[0]):
            labelsi = []
            for j in range(len(x_vector[i])):
                if (x_vector[i][j] in special_character):
                    labelsi.append('O')
                else:
                    if index[i*64+j] == 0:
                        labelsi.append('B_W')
                    elif index[i*64+j] == 1:
                        labelsi.append('I_W')
                    elif index[i*64+j] == 2:
                        labelsi.append('O')
            labels.append(labelsi)
        print (labels[-1])
        # lol
        k = 0
        correct = 0
        all_word = 0
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                all_word += 1
                if(labels[i][j] == y_vector[i][j]):
                    correct +=1
                else:
                    if(y_vector[i][j] == 'O'):
                        k+=1
                        print ( x_real_vector[i][j] ,',', labels[i][j], pred[i*64+j] , ',', y_vector[i][j])
        print (k)
        print (correct/all_word)
        print (all_word)
        print (correct)
    #     corr_pred = []
    #     for i in range(len(x_one_hot_vector)):
    #         corr_pred.append(index[i])
    #     labels = []
    #     for j in range(len(x_test_raw)):
    #         if (corr_pred[j] == 0):
    #             labels.append([x_test_raw[j],'B_W'])
    #         elif (corr_pred[j] == 1):
    #             labels.append([x_test_raw[j],'I_W'])
    #         else:
    #             labels.append([x_test_raw[j],'O'])
    #     # print (tokens)
    #     all_tokens.append(tokens)
    # all_tokens = np.asarray(all_tokens)
    # y_test_raw = np.asrray(y_test_raw)
    # print (all_tokens.shape)
    # print (y_test_raw.shape)
    # print(all_tokens)
def token_sentence(sentence):
    not_have = []
    all_tokens = []
    x_one_hot_vector = []
    real_word = []
    input_size = 64
    all_single_word = word_tokenize(sentence)
    print (all_single_word)
    # lol
    x_one_hot_vectori = []
    if(len(all_single_word) <= input_size):
        real_word = all_single_word
    number_words = len(word2int)
    number_replace = '1000'
    for i in range(len(all_single_word)):
        if(hasNumbers(all_single_word[i])):
            all_single_word[i] = number_replace
    if(len(all_single_word) <= 64):
        for i in range(len(all_single_word)):
            if all_single_word[i] in word2int:
                x_one_hot_vectori.append(to_one_hot(word2int[all_single_word[i]], number_words)) 
            else:
                not_have.append(all_single_word[i])
                x_one_hot_vectori.append(to_one_hot(word2int[number_replace], number_words))
        for i in range(len(all_single_word), input_size, 1):
            temp = np.zeros(number_words,dtype = np.int8)
            x_one_hot_vectori.append(temp)
    x_one_hot_vector.append(x_one_hot_vectori)
    x_one_hot_vector = np.asarray(x_one_hot_vector)
    print (x_one_hot_vector.shape)
        # lol
    # x_data = [x_one_hot_vector]
    x_data = np.asarray(x_one_hot_vector)
    real_word = np.asarray(real_word)
    print (x_data.shape)
    # lol

    num_units = [32,8]
    embedding_dim = 50
    epochs = 500
    batch_size = 128
    learning_rate = 0.2
    file_to_save_model = 'model_saved_ver5/model' + str(input_size) + '-' + str(num_units) + '-' + str(embedding_dim) + '-' + str(batch_size)+'.meta'

    with tf.Session() as sess:
        
        #First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph(file_to_save_model)
        saver.restore(sess,tf.train.latest_checkpoint('model_saved_ver5/'))
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
        # print (pred)
        index = tf.argmax(pred, axis=1, name=None)
        index = sess.run(index)
        # print (index)
        # print (index.shape)
        # print (real_word.shape)
        labels = []
        for i in range(real_word.shape[0]):
            if (index[i] == 0):
                labels.append([real_word[i],'B_W'])
            elif (index[i] == 1):
                labels.append([real_word[i],'I_W'])
            else :
                labels.append([real_word[i],'O'])
        labels = np.asarray(labels)
        # print (labels.shape)
        tokens = compound_word(labels)
    # print (tokens)
        all_tokens.append(tokens)
    return all_tokens
if __name__ == '__main__':
    # tokenize_corpus()
    # all_tokens = tokenize_corpus()
    # print (all_tokens)
    # with open('tokens_corpus.pkl','wb') as output:
    #         pk.dump(all_tokens,output,pk.HIGHEST_PROTOCOL)
    # # evaluate()
    # sentence = "xem tình trạng tiền trong tài khoản của tôi"
    # all_tokens  = token_sentence(sentence)
    # print (all_tokens)
    evaluate()