import numpy as np
leng = 0
train_file = open('./train/train.txt','w')
test_file = open('./train/test.txt','w')
with open ("./text_classifier_ver2.txt", encoding="utf8") as input:
    for line in input:
        leng = leng +1

train_size = int(leng*0.8)

train = np.random.choice(leng,train_size,replace=False)
train = sorted(list(train))
for i in range(leng):
    if i in train:
        train_file.write(str(i)+" ")
    if i not in train:
        test_file.write(str(i)+" ")
train_file.close()
test_file.close()
with open('./train/train.txt','r') as inp:
            line = inp.readline()
            line = line.strip()
            #for line in inp:
            temp = line.split(" ")
            h = [int(i) for i in temp]

print(h)