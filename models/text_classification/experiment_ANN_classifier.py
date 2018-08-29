from preprocessing_classifier import PreprocessingDataClassifier
from ANN_classifier import Classifier
from multiprocessing import Pool
from queue import Queue
from sklearn.model_selection import ParameterGrid
import pickle as pk
queue = Queue()
# read data to a file
CLASSIFY_BY_SOFTMAX = 1
CLASSIFY_BY_SVM = 2
CLASSIFY_BY_KNN = 3
CLASSIFY_BY_BAYESIAN = 4
OPTIMIZER_BY_GRADIENT = 5
OPTIMIZER_BY_SGD = 6
OPTIMIZER_BY_ADAM = 7
def read_trained_data(file_trained_data):
    with open(file_trained_data,'rb') as input_file :
        vectors = pk.load(input_file)
        word2int = pk.load(input_file)
        int2word = pk.load(input_file)
    return vectors, word2int, int2word
def train_model(batch_size_classifier):
    # batch_size_classifier = item["batch_size_classifier"]
    summary = open("../../results/text_classification/ANN_ver5.csv",'a+')
    summary.write("window_size,Embedding,Batch Size Word2vec,Batch Size Classifier,Accuracy\n")
    
    for window_size in window_sizes:
        for embedding_dim in embedding_dims:
            for batch_size_word2vec in batch_size_word2vecs:
                file_to_save_word2vec_data = '../../results/word2vec/ver5/ws-' + str(window_size) + '-embed-' + str(embedding_dim) + 'batch_size-' + str(batch_size_word2vec) + '.pkl'
                
                vectors, word2int, int2word = read_trained_data(file_to_save_word2vec_data)

                file_to_save_classified_data = '../../results/text_classification/ANN_ver5/ws-' + str(window_size) + '-embed-' + str(embedding_dim) + 'batch_size_w2c-' + str(batch_size_word2vec) + 'batch_size_cl' + str(batch_size_classifier)
                
                classifier = Classifier(vectors, word2int, int2word, input_size, num_classes, window_size, 
                        epoch_classifier ,embedding_dim,batch_size_classifier, optimizer_method,
                        file_to_save_classified_data=file_to_save_classified_data)
                accuracy, int2intent = classifier.train(file_data_classifier)
                summary.write(str(window_size)+","+str(embedding_dim)+","+str(batch_size_word2vec)+","+str(batch_size_classifier)+","+str(accuracy)+"\n")
                # print (int2intent)
window_sizes = [2]
embedding_dims = [32]
batch_size_word2vecs = [8]
file_to_save_word2vec_datas = []
input_size = 16
num_classes = 8
epoch_classifier = 500

file_data_classifier = '../../data/text_classsifier_ver3.txt'
# file_to_save_classified_data = "test.pkl"
optimizer_method = OPTIMIZER_BY_GRADIENT
batch_size_classifiers = [16]
# param_grid = {
#     'batch_size_classifier': batch_size_classifiers,
# }
# for item in list(ParameterGrid(param_grid)) :
#     queue.put_nowait(item)
# Consumer
if __name__ == '__main__':
    # pool = Pool(16)
    # pool.map(train_model, list(queue.queue))
    # pool.close()
    # pool.join()
    # pool.terminate()
    for i in range(len(batch_size_classifiers)):
        train_model(batch_size_classifiers[i])