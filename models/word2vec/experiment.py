from preprocessing_word2vec import Preprocess_W2v
from word2vec import Word2Vec
from multiprocessing import Pool
from queue import Queue
from sklearn.model_selection import ParameterGrid
queue = Queue()
def train_model(window_size, embedding_dim, batch_size_word2vec):
    file_to_save_trained_data = '../../results/word2vec/ver7/ws-' + str(window_size) + '-embed-' + str(embedding_dim) + 'batch_size-' + str(batch_size_word2vec) + '.pkl'
    word2vec = Word2Vec(window_size = window_size, epoch_word2vec = epoch_word2vec, embedding_dim = embedding_dim,
                        batch_size_word2vec = batch_size_word2vec, file_to_save_trained_data = file_to_save_trained_data)
    vectors, word2int, int2word = word2vec.train()
window_size = 2
epoch_word2vec = 20
embedding_dims = [32]
batch_size_word2vecs = [4]



# Consumer
if __name__ == '__main__':
    for batch_size_word2vec in batch_size_word2vecs:
        for embedding_dim in embedding_dims:
            train_model(window_size,embedding_dim,batch_size_word2vec)