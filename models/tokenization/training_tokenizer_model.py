from bidirectional_lstm_tokenizer import BiLSTMTokenizer
file_data = '../../data/tokenize/train.txt'
file_to_save_model = '../../results/tokenization/model/'
input_size = 128
num_units = [16,4]
embedding_dim = 50
epochs = 200
batch_size = 32
learning_rate = 0.1
tokenizer = BiLSTMTokenizer(num_units, input_size, embedding_dim, batch_size, epochs, learning_rate, file_data, file_to_save_model)
tokenizer.fit()