from bidirectional_lstm_tokenizer import BiLSTMTokenizer
file_data = '../../data/tokenize/train fix.txt'

input_size = 64
num_units = [32,8]
embedding_dim = 50
epochs = 500
batch_size = 128
learning_rate = 0.2
patience = 20
file_to_save_model = '../../results/tokenization/model_saved_ver3/model' + str(input_size) + '-' + str(num_units) + '-' + str(embedding_dim) + '-' + str(batch_size)
tokenizer = BiLSTMTokenizer(num_units, input_size, embedding_dim, batch_size, epochs, learning_rate,patience ,file_data, file_to_save_model)
tokenizer.fit()