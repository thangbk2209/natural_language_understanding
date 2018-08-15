import plotly.offline as py
import sys
import codecs
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle as pk
import numpy as np
file_to_save_word2vec_data = './ver2/ws-2-embed-32batch_size-4.pkl'
def read_trained_data(file_trained_data):
    with open(file_trained_data,'rb') as input_file :
        vectors = pk.load(input_file)
        word2int = pk.load(input_file)
        int2word = pk.load(input_file)
    return vectors, word2int, int2word
wv, word2int, int2word = read_trained_data(file_to_save_word2vec_data)
print (word2int)
number_of_words = len(word2int)
print (number_of_words)
vocabulary = []
for i in range(number_of_words):
    # print (vectors[i])
    vocabulary.append(int2word[i])

def main():
    # embeddings_file = sys.argv[1]
    # wv, vocabulary = load_embeddings(embeddings_file)
 
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(wv[:number_of_words,:])
 
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()
 

if __name__ == '__main__':
    main()