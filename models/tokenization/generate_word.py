from nltk.tokenize import sent_tokenize, word_tokenize
token_file = open('../../data/tokenize_data.txt','w',encoding="utf8")
special_character = '[@!#$%^&;*()<–>?/\"“”\',|0-9}{~:]'
with open ('../../data/corpus.txt',encoding = 'utf-8') as corpus_file:
    data = corpus_file.read()
    sentences = sent_tokenize(data)
    for i in range(len(sentences)):
        words = word_tokenize(sentences[i])
        for j in range(len(words)):
            if(words[j] in special_character):
                if(j == len(words)-1):
                    print ('true')
                    token_file.write(words[j] + '\t' + 'O' + '\n')
                    token_file.write('\n')
                else:
                    token_file.write(words[j] + '\t' + 'O' + '\n')
            else:
                if(j == len(words)-1):
                    print ('true')
                    token_file.write(words[j] + '\t' + 'B_W' + '\n')
                    token_file.write('\n')
                else:
                    token_file.write(words[j] + '\t' + 'B_W' + '\n')
                