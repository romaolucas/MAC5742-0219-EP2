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

def write_to_output(FO, file_name, times):
    import numpy as np
    FO.write("Media dos tempos de {} : {}\n".format(file_name, np.mean(times)))
    FO.write("Desvio padrao de {} : {}".format(file_name, np.std(times)))

list_of_files = glob.glob('results/*.log')
list_of_images = ['moby_dick', 'king_james_bible', 'hubble_1', 'mercury']

folder = ['results']

file_times = {}
for file_name in list_of_files:
    FI = open(file_name, 'r')
    FO = open(file_name.replace('log', 'out'), 'w') 
    file_times[file_name] = [d['time'] for  d in get_data_list(FI)]
    file = (file_name.split('/')[1]).split('.')[0]
    write_to_output(FO, file_name, file_times[file_name])
    FO.close()
    FI.close()
    axes = plt.gca()
    axes.set_ylim([0,50])
    plt.title(file)
    plt.xlabel('Entrada')
    plt.ylabel('Tempo de Execucao (s)')
    plt.xticks(range(len(list_of_images)), list_of_images, size='small')
    plt.bar(range(len(list_of_images)), file_times[file_name], label=file, color='b')
    plt.savefig(file + '_seq.png')