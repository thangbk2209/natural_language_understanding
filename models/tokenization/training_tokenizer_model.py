from bidirectional_lstm_tokenizer import BiLSTMTokenizer
file_data = '../../data/tokenize/train fix.txt'

input_size = 64
num_units = [32,4]
embedding_dim = 50
epochs = 500
batch_size = 32
learning_rate = 0.2
patience = 20
file_to_save_model = '../../results/tokenization/model_saved_ver11'
print (file_to_save_model)
file_model = open('model_information.txt','w', encoding="utf8")
model_info = file_to_save_model + ":" + str(num_units) + str(input_size) + str(embedding_dim) + str(epochs) + str(batch_size) + str(learning_rate)
file_model.write(model_info)
tokenizer = BiLSTMTokenizer(num_units, input_size, embedding_dim, batch_size, epochs, learning_rate,patience ,file_data, file_to_save_model)
tokenizer.fit()