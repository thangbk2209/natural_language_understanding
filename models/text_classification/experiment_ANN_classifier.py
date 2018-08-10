from preprocessing_classifier import PreprocessingDataClassifier
# read data to a file
    
def read_trained_data(file_trained_data):
    with open(file_trained_data,'rb') as input_file :
        vectors = pk.load(input_file)
        word2int = pk.load(input_file)
        int2word = pk.load(input_file)
    return vectors, word2int, int2word

window_sizes = [1,2]
epoch_word2vec = 2000
embedding_dims = [8, 16, 32, 50, 64, 100, 128]
batch_size_word2vecs = [4, 8, 16, 32, 64]
file_to_save_word2vec_datas = []
input_size = 16
file_data_classifier = '../../data/vietnam.txt'
for window_size in window_sizes:
    for embedding_dim in embedding_dims:
        for batch_size_word2vec in batch_size_word2vecs:
            file_to_save_word2vec_datas = '../../results/word2vec/ws-' + str(window_size) + '-embed-' + embedding_dim + 'batch_size-' + str(batch_size_word2vec) + '.pkl'
            vectors, word2int, int2word = read_trained_data(file_to_save_word2vec_data)
            preprocessing_data_classifier = PreprocessingDataClassifier(vectors, embedding_dim, input_size, file_data_classifier, word2int, int2word)
            preprocessing_data_classifier.preprocessing_data()