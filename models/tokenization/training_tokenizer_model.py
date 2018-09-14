from bidirectional_lstm_tokenizer import BiLSTMTokenizer
file_data = '../../data/tokenize/train.txt'
file_to_save_model = '../../results/tokenization/model/'
input_size = 128
num_units = [3]
embedding_dim = 50
epochs = 2
batch_size = 2
learning_rate = 0.02
tokenizer = BiLSTMTokenizer(num_units, input_size, embedding_dim, batch_size, epochs, learning_rate, file_data, file_to_save_model)
tokenizer.fit()