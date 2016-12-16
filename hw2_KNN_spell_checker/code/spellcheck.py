# Zhiping X. 2016-10-05 22:38:48 -0500
from helper import *
import time
import sys
import matplotlib.pyplot as plt

# Common Usage:
# python spellcheck.py 'spellcheck_input.txt' '3esl.txt'
# check out Answer.pdf to see what else can spellcheck.py do
# See file reader, writer, and other practical function in helper.py
# spellcheck.py is only for ML logic
def spellcheck(checked_file='spellcheck_input.txt', dictionary='3esl.txt', output_to='corrected.txt'):
    dictionarywords = split_dictionary_by_alphabet(load_dictionary(dictionary))
    find_closest_word('tincker', dictionarywords)
    with open(output_to, 'w') as output_f:
        with open(checked_file, 'r') as f:
            current = ''
            word_flag = True
            while True:
                c = f.read(1).lower()
                if c:
                    is_word = 0 <= ord(c)-97 < 26
                    if word_flag == is_word:
                        current += c
                    else:
                        if word_flag:
                            output_f.write(find_closest_word_fast(current, dictionarywords, 2, 1, 4))
                            print current, qwerty_find_closest_word_fast(current, dictionarywords)
                            # print current, find_closest_word_fast(current, dictionarywords, 2, 1, 4)
                        else:
                            output_f.write(current)
                            # print current
                        current = c
                        word_flag = is_word
                else:
                    break
        output_f.write('\n\n')
    print 'result saved to ' + output_to + ' Don\'t depend on this...'


###
def levenshtein_distance(str1, str2, dele_cost=1, inser_cost=1, subs_cost=1):
    d = ([[0] + [i*inser_cost for i in range(1, len(str2)+1)]] +
            [[j * dele_cost]+[0 for i in range(len(str2))]
                            for j in range(1, len(str1)+1)])
    for j in range(1, len(str2)+1):
        for i in range(1, len(str1)+1):
            if str1[i-1] == str2[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min( [d[i-1][j]   + dele_cost,
                                d[i][j-1]   + inser_cost,
                                d[i-1][j-1] + subs_cost])
    return d[len(str1)][len(str2)]

def qwerty_levenshtein_distance(str1, str2, deletion_cost=1, insertion_cost=1):
    d = ([[i*insertion_cost for i in range(len(str2)+1)]] +
            [[j * deletion_cost]+[0 for i in range(len(str2))]
                            for j in range(1, len(str1)+1)])
    for j in range(1, len(str2)+1):
        for i in range(1, len(str1)+1):
            if str1[i-1] == str2[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min( [d[i-1][j]   + deletion_cost,
                                d[i][j-1]   + insertion_cost,
                                d[i-1][j-1] + qwerty_substitution_cost(str1[i-1], str2[j-1])])
    return d[len(str1)][len(str2)]


###
def find_closest_word(str1, dictionary):
    closest_word = dictionary[0]
    min_distance = levenshtein_distance(str1, closest_word)
    for str2 in dictionary:
        curr_distance = levenshtein_distance(str1, str2)
        # if curr_distance == 0:
        #     return str2
        if curr_distance < min_distance:
            min_distance = curr_distance
            closest_word = str2
    return closest_word

def find_closest_word_fast(str1, dictionary, dele_cost=1, inser_cost=1, subs_cost=1):
    # print dictionary[1]
    dictionary = dictionary[ord(str1[0].lower())-97]

    # print str1, dictionary, ord(str1[0].lower())-97,
    closest_word = dictionary[0]
    min_distance = levenshtein_distance(str1, closest_word, dele_cost, inser_cost, subs_cost)
    for str2 in dictionary:
        curr_distance = levenshtein_distance(str1, str2, dele_cost, inser_cost, subs_cost)
        if curr_distance == 0:
            return str2
        if curr_distance < min_distance:
            min_distance = curr_distance
            closest_word = str2
    return closest_word

def qwerty_find_closest_word_fast(str1, dictionary, dele_cost=1, inser_cost=1):
    # print dictionary[1]
    dictionary = dictionary[ord(str1[0].lower())-97]

    # print str1, dictionary, ord(str1[0].lower())-97,
    closest_word = dictionary[0]
    min_distance = qwerty_levenshtein_distance(str1, closest_word, dele_cost, inser_cost)
    for str2 in dictionary:
        curr_distance = qwerty_levenshtein_distance(str1, str2, dele_cost, inser_cost)
        if curr_distance == 0:
            return str2
        if curr_distance < min_distance:
            min_distance = curr_distance
            closest_word = str2
    return closest_word


###
def measure_error(typos, truewords, dictionarywords, dele_cost=1, inser_cost=1, subs_cost=1):
    error_count = 0
    for i in range(len(typos)):
        try:
        # print typos[i], typos[i][0], ord(typos[i][0].lower())-97
        # if 0 <= ord(typos[i][0].lower())-97 <= 26:
            corrected = find_closest_word_fast(
                            typos[i],
                            split_dictionary_by_alphabet(dictionarywords),
                            dele_cost, inser_cost, subs_cost)
            if corrected != truewords[i]:
                error_count += 1
        except:
            error_count += 1
            pass
        #     print i, typos[i], truewords[i], corrected, '--' ,error_count
        # else:
        #     print i, typos[i], truewords[i], corrected
    return 1. * error_count / len(typos)

def qwerty_measure_error(typos, truewords, dictionarywords, dele_cost=1, inser_cost=1):
    error_count = 0
    for i in range(len(typos)):
        # corrected = find_closest_word(typos[i], dictionarywords)
        try:
            corrected = qwerty_find_closest_word_fast(
                            typos[i],
                            split_dictionary_by_alphabet(dictionarywords),
                            dele_cost, inser_cost)
            if corrected != truewords[i]:
                error_count += 1
        except:
            error_count += 1
            pass
        #     print i, typos[i], truewords[i], corrected, '--' ,error_count
        # else:
        #     print i, typos[i], truewords[i], corrected
    return 1. * error_count / len(typos)


###
def try_best_cost_combination(typo_file='wikipediatypocleaner.txt', dictionary_file='3esl.txt', output_to='best_cost_combination_record.txt'):
    typos, truewords = load_typo_tuple(typo_file)
    dictionarywords = load_dictionary(dictionary_file)
    start = time.time()
    # print measure_error(typos, truewords, dictionarywords, 1, 1, 1), time.time()-start
    with open(output_to, 'w') as f:
        for i in [0, 1, 2, 4]:
            for j in [0, 1, 2, 4]:
                for k in [0, 1, 2, 4]:
                    err = measure_error(typos, truewords, dictionarywords, i, j, k)
                    # err = 1.23
                    f.write("%i\t%i\t%i\t%f\n" % (i, j, k, err))
                    # f.write(str(i) + '\t' + str(j) + '\t' +  str(k) + '\t' +  str(err))
                    print i, j, k, err, time.time()-start

def qwerty_try_best_cost_combination(typo_file='wikipediatypocleaner.txt', dictionary_file='3esl.txt', output_to='best_qwerty_cost_combination_record.txt'):
    typos, truewords = load_typo_tuple(typo_file)
    dictionarywords = load_dictionary(dictionary_file)
    start = time.time()
    # print measure_error(typos, truewords, dictionarywords, 1, 1, 1), time.time()-start
    with open(output_to, 'w') as f:
        for i in [0, 1, 2, 4]:
            for j in [0, 1, 2, 4]:
                err = qwerty_measure_error(typos, truewords, dictionarywords, i, j)
                # err = 1.23
                f.write("%i\t%i\t%f\n" % (i, j, err))
                # f.write(str(i) + '\t' + str(j) + '\t' +  str(k) + '\t' +  str(err))
                print i, j, err, time.time()-start

# Precalculated distance table by get_keyboard_distance_table()
# See also helper.py
distance_table = [
[0, 5, 3, 2, 3, 3, 4, 5, 8, 6, 7, 8, 7, 6, 9, 10, 1, 4, 1, 5, 7, 4, 2, 2, 6, 1],
[5, 0, 2, 3, 4, 2, 1, 2, 5, 3, 4, 5, 2, 1, 6, 7, 6, 3, 4, 2, 4, 1, 5, 3, 3, 4],
[3, 2, 0, 1, 2, 2, 3, 4, 7, 5, 6, 7, 4, 3, 8, 9, 4, 3, 2, 4, 6, 1, 3, 1, 5, 2],
[2, 3, 1, 0, 1, 1, 2, 3, 6, 4, 5, 6, 5, 4, 7, 8, 3, 2, 1, 3, 5, 2, 2, 2, 4, 3],
[3, 4, 2, 1, 0, 2, 3, 4, 5, 5, 6, 7, 6, 5, 6, 7, 2, 1, 2, 2, 4, 3, 1, 3, 3, 4],
[3, 2, 2, 1, 2, 0, 1, 2, 5, 3, 4, 5, 4, 3, 6, 7, 4, 1, 2, 2, 4, 1, 3, 3, 3, 4],
[4, 1, 3, 2, 3, 1, 0, 1, 4, 2, 3, 4, 3, 2, 5, 6, 5, 2, 3, 1, 3, 2, 4, 4, 2, 5],
[5, 2, 4, 3, 4, 2, 1, 0, 3, 1, 2, 3, 2, 1, 4, 5, 6, 3, 4, 2, 2, 3, 5, 5, 1, 6],
[8, 5, 7, 6, 5, 5, 4, 3, 0, 2, 1, 2, 3, 4, 1, 2, 7, 4, 7, 3, 1, 6, 6, 8, 2, 9],
[6, 3, 5, 4, 5, 3, 2, 1, 2, 0, 1, 2, 1, 2, 3, 4, 7, 4, 5, 3, 1, 4, 6, 6, 2, 7],
[7, 4, 6, 5, 6, 4, 3, 2, 1, 1, 0, 1, 2, 3, 2, 3, 8, 5, 6, 4, 2, 5, 7, 7, 3, 8],
[8, 5, 7, 6, 7, 5, 4, 3, 2, 2, 1, 0, 3, 4, 1, 2, 9, 6, 7, 5, 3, 6, 8, 8, 4, 9],
[7, 2, 4, 5, 6, 4, 3, 2, 3, 1, 2, 3, 0, 1, 4, 5, 8, 5, 6, 4, 2, 3, 7, 5, 3, 6],
[6, 1, 3, 4, 5, 3, 2, 1, 4, 2, 3, 4, 1, 0, 5, 6, 7, 4, 5, 3, 3, 2, 6, 4, 2, 5],
[9, 6, 8, 7, 6, 6, 5, 4, 1, 3, 2, 1, 4, 5, 0, 1, 8, 5, 8, 4, 2, 7, 7, 9, 3, 10],
[10, 7, 9, 8, 7, 7, 6, 5, 2, 4, 3, 2, 5, 6, 1, 0, 9, 6, 9, 5, 3, 8, 8, 10, 4, 11],
[1, 6, 4, 3, 2, 4, 5, 6, 7, 7, 8, 9, 8, 7, 8, 9, 0, 3, 2, 4, 6, 5, 1, 3, 5, 2],
[4, 3, 3, 2, 1, 1, 2, 3, 4, 4, 5, 6, 5, 4, 5, 6, 3, 0, 3, 1, 3, 2, 2, 4, 2, 5],
[1, 4, 2, 1, 2, 2, 3, 4, 7, 5, 6, 7, 6, 5, 8, 9, 2, 3, 0, 4, 6, 3, 1, 1, 5, 2],
[5, 2, 4, 3, 2, 2, 1, 2, 3, 3, 4, 5, 4, 3, 4, 5, 4, 1, 4, 0, 2, 3, 3, 5, 1, 6],
[7, 4, 6, 5, 4, 4, 3, 2, 1, 1, 2, 3, 2, 3, 2, 3, 6, 3, 6, 2, 0, 5, 5, 7, 1, 8],
[4, 1, 1, 2, 3, 1, 2, 3, 6, 4, 5, 6, 3, 2, 7, 8, 5, 2, 3, 3, 5, 0, 4, 2, 4, 3],
[2, 5, 3, 2, 1, 3, 4, 5, 6, 6, 7, 8, 7, 6, 7, 8, 1, 2, 1, 3, 5, 4, 0, 2, 4, 3],
[2, 3, 1, 2, 3, 3, 4, 5, 8, 6, 7, 8, 5, 4, 9, 10, 3, 4, 1, 5, 7, 2, 2, 0, 6, 1],
[6, 3, 5, 4, 3, 3, 2, 1, 2, 2, 3, 4, 3, 2, 3, 4, 5, 2, 5, 1, 1, 4, 4, 6, 0, 7],
[1, 4, 2, 3, 4, 4, 5, 6, 9, 7, 8, 9, 6, 5, 10, 11, 2, 5, 2, 6, 8, 3, 3, 1, 7, 0]]
def qwerty_substitution_cost(c1, c2):
    if 0 <= ord(c1.lower())-97 < 26 and 0 <= ord(c2.lower())-97 < 26:
        return distance_table[ord(c1.lower())-97][ord(c2.lower())-97]
    else:   # If out of range, no longer letter any more. Return biggest cost
        return 12


if __name__ == '__main__':
    spellcheck(sys.argv[1], sys.argv[2])
