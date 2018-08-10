from preprocessing_classifier import PreprocessingDataClassifier
from ANN_classifier import Classifier
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
def train_model(item):
    batch_size_classifier = item["batch_size_classifier"]
    for window_size in window_sizes:
        for embedding_dim in embedding_dims:
            for batch_size_word2vec in batch_size_word2vecs:
                file_to_save_word2vec_data = '../../results/word2vec/ws-' + str(window_size) + '-embed-' + str(embedding_dim) + 'batch_size-' + str(batch_size_word2vec) + '.pkl'
                
                vectors, word2int, int2word = read_trained_data(file_to_save_word2vec_data)

                file_to_save_classified_data = '../../results/ANN/ws-' + str(window_size) + '-embed-' + str(embedding_dim) + 'batch_size_w2c-' + str(batch_size_word2vec) + 'batch_size_cl' + str(batch_size_classifier) + '.pkl'
                
                classifier = Classifier(vectors, word2int, int2word, input_size, num_classes, window_size, 
                        epoch_classifier ,embedding_dim,batch_size_classifier, optimizer_method,
                        file_to_save_classified_data=file_to_save_classified_data)
                classifier.train(file_data_classifier)
window_sizes = [1,2]
epoch_word2vec = 2000
embedding_dims = [8, 16, 32, 50, 64, 100, 128]
batch_size_word2vecs = [4, 8, 16, 32, 64]
file_to_save_word2vec_datas = []

input_size = 16
num_classes = 8
epoch_classifier = 1

file_data_classifier = '../../data/vietnam.txt'
file_to_save_classified_data = "test.pkl"
optimizer_method = OPTIMIZER_BY_GRADIENT
batch_size_classifier = [4,8,16,32]
param_grid = {
    'batch_size_classifier': batch_size_classifiers,
}
for item in list(ParameterGrid(param_grid)) :
    queue.put_nowait(item)
# Consumer
pool = Pool(16)
pool.map(train_model, list(queue.queue))
pool.close()
pool.join()
pool.terminate()
            
            
            
            # preprocessing_data_classifier = PreprocessingDataClassifier(vectors, embedding_dim, input_size, file_data_classifier, word2int, int2word)
            
            # preprocessing_data_classifier.preprocessing_data()