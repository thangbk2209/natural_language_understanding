from preprocessing_classifier import PreprocessingDataClassifier
from KNN import Classifier
from multiprocessing import Pool
from queue import Queue
from sklearn.model_selection import ParameterGrid
import pickle as pk
queue = Queue()
def read_trained_data(file_trained_data):
    with open(file_trained_data,'rb') as input_file :
        vectors = pk.load(input_file)
        word2int = pk.load(input_file)
        int2word = pk.load(input_file)
    return vectors, word2int, int2word
def train_model(n_neighbors):
    summary = open("../../results/text_classification/KNN.csv",'a+')
    summary.write("window_size,Embedding,Batch Size Word2vec,Number of neighbors,Accuracy\n")
    
    for window_size in window_sizes:
        for embedding_dim in embedding_dims:
            for batch_size_word2vec in batch_size_word2vecs:
                file_to_save_word2vec_data = '../../results/word2vec/ver7/ws-' + str(window_size) + '-embed-' + str(embedding_dim) + 'batch_size-' + str(batch_size_word2vec) + '.pkl'   
                vectors, word2int, int2word = read_trained_data(file_to_save_word2vec_data)
                # print (vectors)
                file_to_save_classified_model = '../../results/text_classification/KNN/'+ str(window_size) + '-embed-' + str(embedding_dim) + 'batch_size_w2c-' + str(batch_size_word2vec) + 'n_neighbors' + str(n_neighbors)
                classifier = Classifier(n_neighbors = n_neighbors,vectors = vectors, embedding_dim = embedding_dim, input_size = input_size,
                word2int = word2int, int2word = int2word,file_to_save_classified_model = file_to_save_classified_model)
                classifier.train(file_data_classifier)
n_neighbors = 3
input_size = 16
window_sizes = [2]
embedding_dims = [32]
batch_size_word2vecs = [4,8,16,32]
file_to_save_word2vec_datas = []
file_data_classifier = '../../data/text_classifier_ver5.txt'

if __name__ == '__main__':
    train_model(n_neighbors)