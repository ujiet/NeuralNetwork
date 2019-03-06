import numpy as np
import random 

def preprocess(dataset_name):

    data_list = []
    data = []
    pre_result = {}

    # get list
    file_path = 'Hopfield_dataset\\' + dataset_name
    with open(file_path) as file:
        data_line = list(file)

        for line in data_line:
            temp = list(line)
            for (i, item) in enumerate(temp):
                if item == ' ':
                    temp[i] = -1
                elif item == '1':
                    temp[i] = 1
                else:
                    del temp[i]
            data_list.append(temp)
    data_list.append([])

    # extend list
    temp = []
    for item in data_list:
        if item == []:
            data.append(temp)
            temp = []
        else:
            temp.extend(item)

    pre_result['data'] = data
    pre_result['graph_width'] = len(data_list[0])   # graph width
    pre_result['neuron_num'] = len(data[0])         # neuron number
    return pre_result

def noise(train_name, ratio = 0.25):
    """ test_name: 'Basic_Training.txt' """
    file_path = 'Hopfield_dataset\\' + train_name
    new_test_file_path = 'Hopfield_dataset\\Noised_' + train_name
    new_test_file = open(new_test_file_path, 'w')

    with open(file_path) as original_test_file:
        while True:
            line = original_test_file.readline()
            if line == '':
                break
            elif line != '\n':
                # add noise
                l = len(line)
                for i in range(l):
                    if random.random() < ratio:
                        if line[i] == '1':
                            line = line[:i] + ' ' + line[i+1:]
                        elif line[i] == ' ':
                            line = line[:i] + '1' + line[i+1:]
            new_test_file.write(line)
    new_test_file.close()
    return 'Noised_' + train_name
