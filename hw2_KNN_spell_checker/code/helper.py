# Zhiping X. 2016-10-05 22:38:48 -0500
import csv
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def load(fileName):
    entries = []
    with open(fileName, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            # for key, value in row.items(): # I don't translate string to bool,
            #     if value == 'true':        # I think string more extendable,
            #         row[key] = True        # Please tell me why,
            #     else:                      # if you think I'm wrong.
            #         row[key] = False       # see also DecisionNode
            entries.append(row)
    return entries


# Read typos_truewords tuple from file
# Return [typos, truewords]
def load_typo_tuple(file_name):
    typo_sets = [[], []]
    with open(file_name, 'r') as f:
        for line in f:
            # print line
            [typo, trueword] = line.strip('\n\r').split('\t')
            typo_sets[0].append(typo)
            typo_sets[1].append(trueword)
    return typo_sets


# dictionarywords = load_dictionary('3esl.txt')
# Read file line by line to an array
def load_dictionary(file_name):
    with open(file_name, 'r') as f:
        return f.read().splitlines()


# Used to generate wikipediatypocleaner, witch is similar to wikipediatypoclean
# Except it not only included the word starting with a~c.
def get_full_wikipediatypoclean(typo_file='wikipediatypo.txt', dictionary_file='3esl.txt', output_to='wikipediatypocleaner.txt'):
    typos, truewords = load_typo_tuple(typo_file)
    dictionarywords = load_dictionary(dictionary_file)
    with open(output_to, 'a') as f:
        for i in range(len(typos)):
            if truewords[i] in dictionarywords:
                f.write(typos[i] + '\t' + truewords[i] + '\n')
        # f.write('Hello\n')


def split_dictionary_by_alphabet(dictionary):
    splited_dictionary = [[] for i in range(26)]
    for word in dictionary:
        # print word[0], ord(word[0].lower())-97
        begin_with = ord(word[0].lower())-97
        if begin_with < 26 and begin_with >= 0: # Discard, if it's not begin with a~z
            splited_dictionary[begin_with].append(word)
    return splited_dictionary


def get_keyboard_distance_table():
    loc = [ ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
            ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
            ['z', 'x', 'c', 'v', 'b', 'n', 'm']]
    qwerty_distance_table = [[0 for j in range(26)] for i in range(26)]
    for nc1 in range(26):
        c1 = str(unichr(97+nc1))
        ic1 = [[ix,iy] for ix, row in enumerate(loc) for iy, i in enumerate(row) if i == c1]
        for nc2 in range(26):
            c2 = str(unichr(97+nc2))
            ic2 = [[ix,iy] for ix, row in enumerate(loc) for iy, i in enumerate(row) if i == c2]
            manhattan_dis = abs(ic1[0][0] - ic2[0][0]) + abs(ic1[0][1] - ic2[0][1])
            qwerty_distance_table[nc1][nc2] = manhattan_dis
    return qwerty_distance_table


# def read_best_cost_combination_record(input_f='best_cost_combination_record.txt'):
#     lst = { 0: {0: {0: 0, 1: 0, 2: 0, 4: 0},
#                 1: {0: 0, 1: 0, 2: 0, 4: 0},
#                 2: {0: 0, 1: 0, 2: 0, 4: 0},
#                 4: {0: 0, 1: 0, 2: 0, 4: 0}},
#             1: {0: {0: 0, 1: 0, 2: 0, 4: 0},
#                 1: {0: 0, 1: 0, 2: 0, 4: 0},
#                 2: {0: 0, 1: 0, 2: 0, 4: 0},
#                 4: {0: 0, 1: 0, 2: 0, 4: 0}},
#             2: {0: {0: 0, 1: 0, 2: 0, 4: 0},
#                 1: {0: 0, 1: 0, 2: 0, 4: 0},
#                 2: {0: 0, 1: 0, 2: 0, 4: 0},
#                 4: {0: 0, 1: 0, 2: 0, 4: 0}},
#             4: {0: {0: 0, 1: 0, 2: 0, 4: 0},
#                 1: {0: 0, 1: 0, 2: 0, 4: 0},
#                 2: {0: 0, 1: 0, 2: 0, 4: 0},
#                 4: {0: 0, 1: 0, 2: 0, 4: 0}}}
#     # print '?'
    # with open(input_f) as f:
    #     content = f.readlines()
    #     # print content
    #     for line in content:
    #         line = line.strip().split()
    #         # print line
    #         lst[int(line[0])][int(line[1])][int(line[2])] = float(line[3])
    # return lst


def read_best_cost_combination_record(input_f='best_cost_combination_record.txt'):
    lst = [.0 for i in range(64)]
    legend = ['' for i in range(64)]
    with open(input_f) as f:
        content = f.readlines()
        for i_line in range(64):
            line = content[i_line].strip().split()
            legend[i_line] = 'D:%s I:%s :S%s' % (line[0], line[1], line[2])
            lst[i_line] = float(line[3])
    return lst, legend

def qwerty_read_best_cost_combination_record(input_f='best_qwerty_cost_combination_record.txt'):
    lst = [.0 for i in range(16)]
    legend = ['' for i in range(16)]
    with open(input_f) as f:
        content = f.readlines()
        for i_line in range(16):
            line = content[i_line].strip().split()
            print i_line, line
            legend[i_line] = 'D:%s I:%s' % (line[0], line[1])
            lst[i_line] = float(line[2])
    return lst, legend


# Usage:
# lst, legend = read_best_cost_combination_record()
# plot_bar_chart(lst, legend)
# lst, legend = qwerty_read_best_cost_combination_record()
# plot_bar_chart(lst, legend)
def plot_bar_chart(lst, legend):
    plt.bar(np.arange(0, len(lst), 1), np.array(lst), width=1)
    for a,b in zip(np.arange(0, len(lst), 1), np.array(lst)):
        plt.text(a, b+0.01, str(round(b, 3)))
    plt.xticks(range(len(lst)), legend, rotation=90)
    plt.show()
