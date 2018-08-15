from preprocessing_word2vec import Preprocess_W2v
from word2vec import Word2Vec
from multiprocessing import Pool
from queue import Queue
from sklearn.model_selection import ParameterGrid
queue = Queue()
def train_model(item):
    window_size = item["window_size"]
    embedding_dim = item["embedding_dim"]
    batch_size_word2vec = item["batch_size_word2vec"]
    file_to_save_trained_data = '../../results/word2vec/ver2/ws-' + str(window_size) + '-embed-' + str(embedding_dim) + 'batch_size-' + str(batch_size_word2vec) + '.pkl'
    word2vec = Word2Vec(window_size = window_size, epoch_word2vec = epoch_word2vec, embedding_dim = embedding_dim,
                        batch_size_word2vec = batch_size_word2vec, file_to_save_trained_data = file_to_save_trained_data)
    vectors, word2int, int2word = word2vec.train()
window_sizes = [2]
epoch_word2vec = 2
embedding_dims = [32]
batch_size_word2vecs = [4]
# file_to_save_trained_datas = []
# for window_size in window_sizes:
#     for embedding_dim in embedding_dims:
#         for batch_size_word2vec in batch_size_word2vecs:
#             file_to_save_trained_data = '../../results/word2vec/ws-' + str(window_size) + '-embed-' + str(embedding_dim) + 'batch_size-' + str(batch_size_word2vec) + '.pkl'
#             file_to_save_trained_datas.append(file_to_save_trained_data)

param_grid = {
    'window_size': window_sizes,
    'embedding_dim': embedding_dims,
    'batch_size_word2vec': batch_size_word2vecs,
}

for item in list(ParameterGrid(param_grid)) :
    queue.put_nowait(item)
# Consumer
if __name__ == '__main__':
    pool = Pool(8)
    pool.map(train_model, list(queue.queue))
    pool.close()
    pool.join()
    pool.terminate()
