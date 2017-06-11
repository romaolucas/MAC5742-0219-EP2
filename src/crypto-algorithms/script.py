# encoding=utf8  
import glob
import matplotlib.pyplot as plt

from pylab import *
from parse import *

def get_data_list(FI):
    data = {}
    data_list = []
    for line in FI:
        r = search('{clock}      task-clock (msec)         #    {cpu} CPUs utilized', line)
        if r:
            data['clock'] = float(r['clock']) 
            data['cpu'] = float(r['cpu'])
        r = search('{time} seconds time elapsed', line)
        if r:
            data['time'] = float(r['time'])
            data_list.append(data.copy())
    return data_list

def write_to_output(FO, file_name, times):
    import numpy as np
    FO.write("Media dos tempos de {} : {}\n".format(file_name, np.mean(times)))
    FO.write("Desvio padrao de {} : {}".format(file_name, np.std(times)))

list_of_files = glob.glob('mandelbrot_seq/*.log')
file_times = {}
for file_name in list_of_files:
    FI = open(file_name, 'r')
    FO = open(file_name.replace('log', 'out'), 'w') 
    file_times[file_name] = [d['time'] for  d in get_data_list(FI)]
    write_to_output(FO, file_name, file_times[file_name])
    FI.close()
    FO.close()
    
x = [2**i for i in range(4, 14)] 
for file_name in file_times:
    plt.plot(x, file_times[file_name], label=(file_name.split('/')[1]).split('.')[0])

axes = plt.gca()
axes.set_ylim([0,150])
plt.legend(loc="upper left")
plt.title('Sequencial')
plt.xlabel('Tamanho da Entrada')
plt.ylabel('Tempo de Execucao (s)')
plt.savefig('seq.png')
plt.close()

list_of_images = ['full', 'elephant', 'seahorse', 'triple_spiral']
num_threads = [2**i for i in range(0, 6)]
folders = ['mandelbrot_pth', 'mandelbrot_omp']

for folder in folders:
    for image in list_of_images:
        file_times = {}
        for nthread in num_threads:
            file_name = folder + "/" + image + "_" + str(nthread) + ".log"
            FI = open(file_name, 'r')
            FO = open(file_name.replace('log', 'out'), 'w')
            file_times[file_name] = [d['time'] for d in get_data_list(FI)]
            write_to_output(FO, file_name, file_times[file_name]) 
            FI.close()
            FO.close()
        x = [2**i for i in range(4, 14)] 
        for file_name in file_times:
            plt.plot(x, file_times[file_name], label=((file_name.split('/')[1]).split('.')[0]).split('_')[-1])

        axes = plt.gca()
        handles, labels = axes.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: int(t[0])))
        axes.set_ylim([0,150])
        plt.legend(handles, labels, title='Numero de Threads', loc="upper left")
        plt.title(image.title())
        plt.xlabel('Tamanho da Entrada')
        plt.ylabel('Tempo de Execucao (s)')
        plt.savefig(folder + '_' + image + '.png')
        plt.close()

           
