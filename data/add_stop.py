from nltk.tokenize import sent_tokenize, word_tokenize
data_file = open('text_classifier_ver7_fix.txt','w',encoding="utf8")
with open ('text_classifier_ver7.txt',encoding = 'utf-8') as corpus_file:
    lines = corpus_file.readlines()
    for line in lines: 
        # token_file.write(words[j] + '\t' + 'O' + '\n')      
        if('?' in line):
            data_file.write(line)
        else:
            data_file.write(line.rstrip('\n') + ' .' +'\n')