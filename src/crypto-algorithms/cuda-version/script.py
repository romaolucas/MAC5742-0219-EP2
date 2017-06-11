# encoding=utf8  
import glob
import matplotlib.pyplot as plt

from pylab import *
from parse import *

def get_data_list(FI):
    data = {}
    data_list = []
    for line in FI:
        r = search('         {clock}      task-clock (msec)         #    {cpu} CPUs utilized', line)
        if r:
            data['clock'] = float(r['clock'].replace(',','.')) 
            data['cpu'] = float(r['cpu'].replace(',','.'))
        r = search('{time} seconds time elapsed', line)
        if r:
            data['time'] = float((r['time'].replace(' ','')).replace(',','.'))
            data_list.append(data.copy())
    return data_list

list_of_files = glob.glob('results/*.log')
list_of_images = ['moby_dick', 'king_james_bible', 'hubble_1', 'mercury']

folder = ['results']

file_times = {}
for file_name in list_of_files:
    FI = open(file_name, 'r')
    file_times[file_name] = [d['time'] for  d in get_data_list(FI)]
    file = (file_name.split('/')[1]).split('.')[0]
    FI.close()
    axes = plt.gca()
    handles, labels = axes.get_legend_handles_labels()
    plt.bar(range(len(list_of_images)), file_times[file_name], label=file)
    plt.title(file)
    plt.xlabel('Entrada')
    plt.ylabel('Tempo de Execucao (s)')
    plt.xticks(range(len(list_of_images)), list_of_images, size='small')
    plt.savefig(file + '.png')