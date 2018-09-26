corpus_file = open('corpus.txt','w', encoding="utf8")
with open ('train.txt',encoding = 'utf-8') as acro_file:
    lines = acro_file.readlines()
    x_traini = []
    y_traini = []
    for line in lines:
        # print (line)
        if line == '\n':
            # print(1)
            corpus_file.write('\n')
        else:
            datai = line.rstrip('\n').split('\t')
            corpus_file.write(datai[0] + ' ')