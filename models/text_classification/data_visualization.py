import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
font = {'size'   : 5}

matplotlib.rc('font', **font)
intents_data = []
with open('../../data/text_classifier_ver2.txt', encoding="utf8") as input:
    for line in input :
        # print (line)
        temp = line.split(",",1)
        intents_data.append(temp[0]) #list of label
print (intents_data)
import collections
x = collections.Counter(intents_data)
l = range(len(x.keys()))
plt.bar(l, x.values(), align='center')
plt.xticks(l, x.keys())
# tick.label.set_fontsize(14) 
# plt.show()
plt.savefig('../../results/text_classification/visulization_data_ver2.png')
# plt.hist(intents_data)
# plt.title("Data visualization")
# plt.xlabel("Intents")
# plt.ylabel("Frequency")

# fig = plt.gcf()

# plot_url = py.plot_mpl(fig, filename='mpl-basic-histogram')