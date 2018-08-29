from preprocessing_word2vec import Preprocess_W2v
from word2vec import Word2Vec
from multiprocessing import Pool
from queue import Queue
from sklearn.model_selection import ParameterGrid
queue = Queue()
def train_model(window_size, embedding_dim, batch_size_word2vec):
    file_to_save_trained_data = '../../results/word2vec/ver5/ws-' + str(window_size) + '-embed-' + str(embedding_dim) + 'batch_size-' + str(batch_size_word2vec) + '.pkl'
    word2vec = Word2Vec(window_size = window_size, epoch_word2vec = epoch_word2vec, embedding_dim = embedding_dim,
                        batch_size_word2vec = batch_size_word2vec, file_to_save_trained_data = file_to_save_trained_data)
    vectors, word2int, int2word = word2vec.train()
window_size = 2
epoch_word2vec = 200
embedding_dims = [32]
batch_size_word2vecs = [16]
# file_to_save_trained_datas = []
# for window_size in window_sizes:
#     for embedding_dim in embedding_dims:
#         for batch_size_word2vec in batch_size_word2vecs:
#             file_to_save_trained_data = '../../results/word2vec/ws-' + str(window_size) + '-embed-' + str(embedding_dim) + 'batch_size-' + str(batch_size_word2vec) + '.pkl'
#             file_to_save_trained_datas.append(file_to_save_trained_data)




# Consumer
if __name__ == '__main__':
    for batch_size_word2vec in batch_size_word2vecs:
        for embedding_dim in embedding_dims:
            train_model(window_size,embedding_dim,batch_size_word2vec)